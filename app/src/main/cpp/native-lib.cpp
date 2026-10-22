#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "llama.h"

namespace {
std::mutex g_mutex;
std::mutex g_load_log_mutex;
std::string g_load_log;
std::string g_last_error;
std::atomic<bool> g_abort{false};

static bool llamaAbortCallback(void *) {
    return g_abort.load(std::memory_order_relaxed);
}

static void captureLlamaLog(enum ggml_log_level level, const char *text, void *) {
    if (!text) return;
    {
        std::lock_guard<std::mutex> lock(g_load_log_mutex);
        g_load_log.append(text);
        constexpr size_t kMaxLogBytes = 12000;
        if (g_load_log.size() > kMaxLogBytes) {
            g_load_log.erase(0, g_load_log.size() - kMaxLogBytes);
        }
    }
    int priority = ANDROID_LOG_INFO;
    if (level >= GGML_LOG_LEVEL_ERROR) priority = ANDROID_LOG_ERROR;
    else if (level >= GGML_LOG_LEVEL_WARN) priority = ANDROID_LOG_WARN;
    else if (level <= GGML_LOG_LEVEL_DEBUG) priority = ANDROID_LOG_DEBUG;
    __android_log_write(priority, "QwenOffline", text);
}

static std::string recentLlamaLog() {
    std::lock_guard<std::mutex> lock(g_load_log_mutex);
    return g_load_log;
}

llama_model *g_model = nullptr;
llama_context *g_ctx = nullptr;
llama_sampler *g_sampler = nullptr;
std::string g_model_path;
int g_threads = 4;
uint32_t g_context = 512;
std::vector<llama_token> g_cached_prompt_tokens;

int choose_threads(int requested) {
    unsigned int c = std::thread::hardware_concurrency();
    int cap = std::min(4, c ? static_cast<int>(c) : 4);
    return requested <= 0 ? std::max(1, cap) : std::clamp(requested, 1, cap);
}

std::string js(JNIEnv *env, jstring value) {
    if (!value) return {};
    const char *c = env->GetStringUTFChars(value, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(value, c);
    return out;
}

size_t rss() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream in(line.substr(7));
            size_t kb = 0;
            in >> kb;
            return kb * 1024;
        }
    }
    return 0;
}

void clear_prompt_cache_locked() {
    g_cached_prompt_tokens.clear();
}

void unload_locked() {
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_model_path.clear();
    clear_prompt_cache_locked();
}

bool reset_sampler_locked(float temp = 0.7f, float top_p = 0.9f, float penalty = 1.1f) {
    if (!g_model || !g_ctx) return false;
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!g_sampler) return false;

    if (temp <= 0.01f) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_greedy());
    } else {
        if (penalty > 1.0f) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(llama_model_get_vocab(g_model)), 64, penalty, 0.0f, 0.0f));
        }
        if (top_p < 1.0f) {
            llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
        }
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    return true;
}

bool recreate_context_locked() {
    if (!g_model) return false;
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    clear_prompt_cache_locked();

    auto p = llama_context_default_params();
    p.n_ctx = g_context;
    p.n_batch = std::min<uint32_t>(g_context, 32);
    p.n_ubatch = std::min<uint32_t>(g_context, 16);
    p.abort_callback = [](void *) -> bool {
        return g_abort.load(std::memory_order_relaxed);
    };
    p.abort_callback_data = nullptr;
    p.n_threads = g_threads;
    p.n_threads_batch = g_threads;
    p.no_perf = true;

    g_ctx = llama_init_from_model(g_model, p);
    if (!g_ctx) return false;
    return reset_sampler_locked();
}

struct Turn {
    std::string role;
    std::string content;
};

std::vector<Turn> parse_history(const std::string &encoded) {
    std::vector<Turn> out;
    if (encoded.empty()) return out;
    size_t start = 0;
    while (start < encoded.size()) {
        size_t end = encoded.find('\x1e', start);
        size_t stop = end == std::string::npos ? encoded.size() : end;
        size_t sep = encoded.find('\x1f', start);
        if (sep != std::string::npos && sep < stop) {
            std::string role = encoded.substr(start, sep - start);
            std::string content = encoded.substr(sep + 1, stop - sep - 1);
            if ((role == "user" || role == "assistant") && !content.empty()) {
                out.push_back({role, content});
            }
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

bool format_chat(const std::string &system_prompt,
                 const std::string &encoded_history,
                 const std::string &user,
                 std::string &out) {
    if (!g_model || user.empty()) return false;

    std::vector<Turn> turns;
    if (!system_prompt.empty()) {
        turns.push_back({"system", system_prompt});
    }

    auto h = parse_history(encoded_history);
    turns.insert(turns.end(), h.begin(), h.end());
    turns.push_back({"user", user});

    std::vector<llama_chat_message> messages;
    messages.reserve(turns.size());
    for (auto &t : turns) messages.push_back({t.role.c_str(), t.content.c_str()});

    const char *tmpl = llama_model_chat_template(g_model, nullptr);
    if (tmpl && *tmpl) {
        int32_t n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
        if (n >= 0) {
            std::vector<char> buffer(static_cast<size_t>(n) + 1, '\0');
            int32_t written = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                        buffer.data(), static_cast<int32_t>(buffer.size()));
            if (written >= 0) {
                out.assign(buffer.data(), static_cast<size_t>(written));
                return true;
            }
        }
    }

    // Default ChatML format for Qwen and modern open-weights
    std::ostringstream fallback;
    if (!system_prompt.empty()) {
        fallback << "<|im_start|>system\n" << system_prompt << "<|im_end|>\n";
    }
    for (auto &t : turns) {
        if (t.role != "system") {
            fallback << "<|im_start|>" << t.role << "\n" << t.content << "<|im_end|>\n";
        }
    }
    fallback << "<|im_start|>assistant\n";
    out = fallback.str();
    return true;
}

void notify_token(JNIEnv *env, jobject self, const std::string &text) {
    jclass cls = env->GetObjectClass(self);
    if (!cls) return;
    jmethodID method = env->GetMethodID(cls, "onNativeToken", "(Ljava/lang/String;)V");
    if (method) {
        jstring value = env->NewStringUTF(text.c_str());
        if (value) {
            env->CallVoidMethod(self, method, value);
            env->DeleteLocalRef(value);
        }
    }
    env->DeleteLocalRef(cls);
}

std::string generate_tokens(const std::string &prompt,
                            int limit,
                            float temp,
                            float top_p,
                            float penalty,
                            JNIEnv *env = nullptr,
                            jobject callback = nullptr,
                            int *generated_tokens_out = nullptr) {
    g_abort.store(false, std::memory_order_relaxed);
    if (!g_model || !g_ctx || !reset_sampler_locked(temp, top_p, penalty)) return {};
    llama_set_abort_callback(g_ctx, llamaAbortCallback, nullptr);

    auto *vocab = llama_model_get_vocab(g_model);
    auto *memory = llama_get_memory(g_ctx);
    if (!vocab || !memory) return {};

    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) return {};
    if (static_cast<uint32_t>(n_prompt + limit + 2) >= g_context) return {};

    std::vector<llama_token> tokens(static_cast<size_t>(n_prompt));
    int tokenized = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (tokenized < 0) return {};

    size_t common = 0;
    while (common < g_cached_prompt_tokens.size() &&
           common < tokens.size() &&
           g_cached_prompt_tokens[common] == tokens[common]) {
        ++common;
    }

    // The app serializes chat history explicitly. Reset the KV cache per request so
    // generated tokens from the previous turn can never become mismatched state.
    llama_memory_clear(memory, true);
    constexpr size_t kPromptChunk = 32;
    for (size_t offset = 0; offset < tokens.size(); offset += kPromptChunk) {
        if (g_abort.load(std::memory_order_relaxed)) return {};
        const size_t chunk = std::min(kPromptChunk, tokens.size() - offset);
        auto batch = llama_batch_get_one(tokens.data() + offset, static_cast<int32_t>(chunk));
        if (llama_decode(g_ctx, batch) != 0) return {};
    }
    g_cached_prompt_tokens.clear();

    std::string result;
    result.reserve(static_cast<size_t>(limit) * 4);
    int generated_tokens = 0;
    for (int i = 0; i < limit; ++i) {
        if (g_abort.load(std::memory_order_relaxed)) break;

        llama_token token = llama_sampler_sample(g_sampler, g_ctx, -1);
        if (token < 0 || llama_vocab_is_eog(vocab, token)) break;

        llama_sampler_accept(g_sampler, token);
        ++generated_tokens;

        char piece_buffer[512];
        int n = llama_token_to_piece(vocab, token, piece_buffer, sizeof(piece_buffer), 0, true);
        std::string piece = n > 0 ? std::string(piece_buffer, n) : std::string();
        result += piece;

        if (env && callback && !piece.empty()) notify_token(env, callback, piece);

        if (static_cast<uint32_t>(tokens.size() + i + 2) >= g_context) break;

        llama_token next = token;
        auto batch = llama_batch_get_one(&next, 1);
        if (llama_decode(g_ctx, batch) != 0) break;
    }
    if (generated_tokens_out) *generated_tokens_out = generated_tokens;
    return result;
}

jstring to_jstring(JNIEnv *env, const std::string &value) {
    return env->NewStringUTF(value.c_str());
}

std::string model_summary_locked() {
    if (!g_model) return "No GGUF model loaded";
    auto *vocab = llama_model_get_vocab(g_model);
    const char *tmpl = llama_model_chat_template(g_model, nullptr);
    std::ostringstream out;
    out << "Model: " << g_model_path
        << "\nSize: " << (llama_model_size(g_model) / (1024 * 1024)) << " MB"
        << "\nParams: " << (llama_model_n_params(g_model) / 1000000) << "M"
        << "\nContext: " << (g_ctx ? llama_n_ctx(g_ctx) : g_context)
        << "\nCPU threads: " << (g_ctx ? llama_n_threads(g_ctx) : g_threads)
        << "\nVocab: " << (vocab ? llama_vocab_n_tokens(vocab) : 0)
        << "\nChat template: " << (tmpl && *tmpl ? "embedded GGUF" : "ChatML fallback")
        << "\nWeights: memory-mapped (no mlock)"
        << "\nCached KV tokens: " << g_cached_prompt_tokens.size()
        << "\nRSS memory: " << (rss() / (1024 * 1024)) << " MB";
    return out.str();
}
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeEngineInfo(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    llama_backend_init();
    std::ostringstream out;
    out << "llama.cpp: " << (llama_version() ? llama_version() : "unknown")
        << "\nABI: armeabi-v7a (ARMv7 32-bit)"
        << "\nTarget: Android 11 / API 30"
        << "\nAdaptive threads: " << choose_threads(0)
        << "\nWeights load mode: LLAMA_LOAD_MODE_MMAP"
        << "\nDefault system prompt: DISABLED (pure injection only)";
    return to_jstring(env, out.str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeLoadModel(JNIEnv *env, jobject, jstring path, jint ctx, jint threads) {
    std::lock_guard<std::mutex> lock(g_mutex);
    llama_backend_init();

    std::string model_path = js(env, path);
    g_last_error.clear();
    {
        std::lock_guard<std::mutex> log_lock(g_load_log_mutex);
        g_load_log.clear();
    }
    llama_log_set(captureLlamaLog, nullptr);
    if (model_path.empty()) {
        g_last_error = "No model file path was selected.";
        return JNI_FALSE;
    }
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file) {
        g_last_error = "Cannot open model file: " + model_path + "\nCheck that the selected file still exists and is readable.";
        return JNI_FALSE;
    }
    model_file.seekg(0, std::ios::end);
    const auto model_bytes = model_file.tellg();
    model_file.close();
    if (model_bytes <= 0) {
        g_last_error = "Model file is empty or could not be measured: " + model_path;
        return JNI_FALSE;
    }

    const int requested_threads = choose_threads(threads);
    const uint32_t requested_context = static_cast<uint32_t>(std::clamp(ctx, 256, 2048));

    if (g_model && g_ctx && g_model_path == model_path &&
        g_threads == requested_threads && g_context == requested_context) {
        g_abort.store(false, std::memory_order_relaxed);
        return JNI_TRUE;
    }

    unload_locked();

    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_extra_bufts = false;
    params.load_mode = LLAMA_LOAD_MODE_MMAP;

    g_model = llama_model_load_from_file(model_path.c_str(), params);
    if (!g_model) {
        std::ostringstream error;
        error << "llama.cpp could not load this GGUF.\nFile size: " << model_bytes << " bytes\nPath: " << model_path;
        const std::string log = recentLlamaLog();
        if (!log.empty()) error << "\n\nNative loader details:\n" << log;
        else error << "\n\nNo native loader details were returned. This can happen if the 32-bit process runs out of address space or memory while mapping the model.";
        g_last_error = error.str();
        return JNI_FALSE;
    }

    g_threads = requested_threads;
    g_context = requested_context;
    if (!recreate_context_locked()) {
        const std::string log = recentLlamaLog();
        g_last_error = "Model weights loaded, but llama.cpp could not create the inference context (requested " + std::to_string(requested_context) + " tokens).";
        if (!log.empty()) g_last_error += "\n\nNative details:\n" + log;
        unload_locked();
        return JNI_FALSE;
    }

    g_model_path = model_path;
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeUnloadModel(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeModelInfo(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return to_jstring(env, model_summary_locked());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeLastError(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return to_jstring(env, g_last_error);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeMemoryBytes(JNIEnv *, jobject) {
    return static_cast<jlong>(rss());
}

extern "C" JNIEXPORT void JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeCancelGeneration(JNIEnv *, jobject) {
    g_abort.store(true, std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeGenerateChatStreaming(
    JNIEnv *env, jobject self,
    jstring system_prompt_j,
    jstring history_encoded_j,
    jstring user_text_j,
    jint max_tokens,
    jfloat temperature,
    jfloat top_p,
    jfloat repeat_penalty) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        notify_token(env, self, "\x01ERROR: No GGUF model loaded");
        notify_token(env, self, "\x01DONE");
        return;
    }

    std::string system_prompt = js(env, system_prompt_j);
    std::string history_encoded = js(env, history_encoded_j);
    std::string user = js(env, user_text_j);
    if (user.empty()) {
        notify_token(env, self, "\x01DONE");
        return;
    }

    std::string formatted;
    if (!format_chat(system_prompt, history_encoded, user, formatted)) {
        notify_token(env, self, "\x01ERROR: Chat template formatting failed");
        notify_token(env, self, "\x01DONE");
        return;
    }

    const int limit = std::clamp(static_cast<int>(max_tokens), 1, 512);
    std::string generated = generate_tokens(formatted, limit, temperature, top_p, repeat_penalty, env, self);

    const bool cancelled = g_abort.load(std::memory_order_relaxed);
    if (generated.empty() && !cancelled) {
        notify_token(env, self, "\x01ERROR: Generation failed");
    }
    notify_token(env, self, "\x01DONE");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qwen_offline_LlamaNativeEngine_nativeBenchmark(JNIEnv *env, jobject, jstring system_j, jint max_tokens) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) return to_jstring(env, "Load a GGUF model before running benchmark.");

    std::string system = js(env, system_j);
    std::vector<std::string> tests = {
        "Say hello in one short sentence.",
        "What is 2 + 2? Answer briefly.",
        "Explain in one sentence what an offline AI model does."
    };

    std::ostringstream out;
    out << "Benchmark: " << g_model_path << "\n";
    out << "Threads: " << g_threads << " | Context: " << g_context << "\n";

    for (size_t i = 0; i < tests.size(); ++i) {
        std::string formatted;
        if (!format_chat(system, "", tests[i], formatted)) continue;

        int tokens_gen = 0;
        const auto started = std::chrono::steady_clock::now();
        std::string ans = generate_tokens(formatted, std::clamp(static_cast<int>(max_tokens), 1, 64),
                                          0.0f, 1.0f, 1.0f, nullptr, nullptr, &tokens_gen);
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        const double tok_s = elapsed_ms > 0 ? (tokens_gen * 1000.0 / elapsed_ms) : 0.0;

        out << "\n[Prompt " << (i + 1) << "] tokens=" << tokens_gen
            << ", time=" << elapsed_ms << "ms (" << tok_s << " tok/s)\n"
            << ans << "\n";
    }
    return to_jstring(env, out.str());
}
