package com.qwen.offline

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URLEncoder
import java.net.URL

/** Small, keyless public-web search client. Search runs only for current/future questions. */
object WebResearch {
    data class Result(val title: String, val url: String, val snippet: String)

    suspend fun search(query: String): List<Result> = withContext(Dispatchers.IO) {
        val cleaned = query.trim().take(240)
        require(cleaned.isNotBlank()) { "Search query is empty" }
        val encoded = URLEncoder.encode(cleaned, "UTF-8")
        val connection = (URL("https://html.duckduckgo.com/html/?q=$encoded").openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 10000
            readTimeout = 12000
            setRequestProperty("User-Agent", "Mozilla/5.0 (Android; Mobile) QwenOffline/1.1")
            setRequestProperty("Accept", "text/html")
        }
        try {
            val status = connection.responseCode
            if (status !in 200..299) throw IllegalStateException("Web search returned HTTP $status")
            val html = connection.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText().take(1_000_000) }
            val blocks = Regex("(?s)<div[^>]*class=\\\"result[^\\\"]*\\\"[^>]*>(.*?)(?=<div[^>]*class=\\\"result|$)", RegexOption.IGNORE_CASE)
            val titlePattern = Regex("(?s)<a[^>]*class=\\\"result__a\\\"[^>]*href=\\\"([^\\\"]+)\\\"[^>]*>(.*?)</a>", RegexOption.IGNORE_CASE)
            val snippetPattern = Regex("(?s)<(?:a|div)[^>]*class=\\\"result__snippet\\\"[^>]*>(.*?)</(?:a|div)>", RegexOption.IGNORE_CASE)
            blocks.findAll(html).mapNotNull { match ->
                val block = match.groupValues[1]
                val titleMatch = titlePattern.find(block) ?: return@mapNotNull null
                val rawUrl = titleMatch.groupValues[1].replace("&amp;", "&")
                val actualUrl = Regex("uddg=([^&]+)").find(rawUrl)?.groupValues?.get(1)?.let {
                    java.net.URLDecoder.decode(it, "UTF-8")
                } ?: rawUrl
                if (!actualUrl.startsWith("https://") && !actualUrl.startsWith("http://")) return@mapNotNull null
                val title = htmlText(titleMatch.groupValues[2]).take(180)
                val snippet = snippetPattern.find(block)?.groupValues?.get(1)?.let(::htmlText)?.take(450).orEmpty()
                if (title.isBlank() || snippet.isBlank()) null else Result(title, actualUrl, snippet)
            }.distinctBy { it.url }.take(4).toList()
        } finally {
            connection.disconnect()
        }
    }

    private fun htmlText(value: String): String = value
        .replace(Regex("<[^>]*>"), " ")
        .replace("&amp;", "&").replace("&quot;", "\"").replace("&#39;", "'")
        .replace("&lt;", "<").replace("&gt;", ">")
        .replace(Regex("&#(\\d+);")) { it.groupValues[1].toIntOrNull()?.toChar()?.toString() ?: " " }
        .replace(Regex("\\s+"), " ").trim()

    fun needsFreshInfo(query: String): Boolean = Regex(
        "\\b(current|currently|today|tonight|now|latest|newest|recent|this week|this month|this year|as of|news|weather|price|stock|who is the current|future|upcoming|next (week|month|year|election|release)|forecast|scheduled|will .{1,30} happen|when will)\\b",
        RegexOption.IGNORE_CASE
    ).containsMatchIn(query)
}
