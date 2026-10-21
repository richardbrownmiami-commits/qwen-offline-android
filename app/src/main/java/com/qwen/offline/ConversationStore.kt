package com.qwen.offline

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class ConversationStore(context: Context) : SQLiteOpenHelper(context, "qwen_chat.db", null, 1) {

    data class Message(val id: Long, val isUser: Boolean, val text: String, val timestamp: Long)

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                is_user INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS messages")
        onCreate(db)
    }

    fun addMessage(isUser: Boolean, text: String): Long {
        val values = ContentValues().apply {
            put("is_user", if (isUser) 1 else 0)
            put("content", text)
            put("created_at", System.currentTimeMillis())
        }
        return writableDatabase.insert("messages", null, values)
    }

    fun getRecentMessages(limit: Int = 30): List<Message> {
        val list = mutableListOf<Message>()
        readableDatabase.rawQuery(
            "SELECT id, is_user, content, created_at FROM messages ORDER BY id DESC LIMIT ?",
            arrayOf(limit.toString())
        ).use { cursor ->
            while (cursor.moveToNext()) {
                list.add(
                    Message(
                        id = cursor.getLong(0),
                        isUser = cursor.getInt(1) == 1,
                        text = cursor.getString(2),
                        timestamp = cursor.getLong(3)
                    )
                )
            }
        }
        return list.reversed()
    }

    fun clearAll() {
        writableDatabase.delete("messages", null, null)
    }
}
