package com.qwen.offline

class LlamaNativeEngine {
    external fun nativeEngineInfo(): String
    external fun nativeLoadModel(path: String, contextSize: Int, threads: Int): Boolean
    external fun nativeUnloadModel()
    external fun nativeModelInfo(): String
    external fun nativeMemoryBytes(): Long
    external fun nativeCancelGeneration()
    external fun nativeGenerateChatStreaming(
        systemPrompt: String,
        historyEncoded: String,
        userText: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        repeatPenalty: Float
    )
    external fun nativeBenchmark(systemPrompt: String, maxTokens: Int): String

    @Volatile
    private var tokenListener: ((String) -> Unit)? = null

    fun onNativeToken(token: String) {
        tokenListener?.invoke(token)
    }

    fun generateChatStreaming(
        systemPrompt: String,
        historyEncoded: String,
        userText: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        repeatPenalty: Float,
        onToken: (String) -> Unit
    ) {
        tokenListener = onToken
        try {
            nativeGenerateChatStreaming(
                systemPrompt,
                historyEncoded,
                userText,
                maxTokens,
                temperature,
                topP,
                repeatPenalty
            )
        } finally {
            tokenListener = null
        }
    }

    fun isModelLoaded(): Boolean = !nativeModelInfo().startsWith("No GGUF model loaded")

    fun encodeHistory(turns: List<Turn>): String = turns.joinToString("\u001e") { turn ->
        val role = if (turn.isUser) "user" else "assistant"
        val content = turn.text.replace("\u001e", " ").replace("\u001f", " ")
        "$role\u001f$content"
    }

    data class Turn(val isUser: Boolean, val text: String)

    companion object {
        init {
            System.loadLibrary("qwenoffline")
        }
        const val DEFAULT_CONTEXT_SIZE = 512
        const val DEFAULT_THREADS = 4
        const val DEFAULT_MAX_TOKENS = 128
        const val DEFAULT_TEMP = 0.7f
        const val DEFAULT_TOP_P = 0.9f
        const val DEFAULT_PENALTY = 1.1f
    }
}
