package com.qwen.offline

import android.content.Context
import android.os.BatteryManager
import java.io.File

object DeviceVitals {

    fun getMemInfo(): MemInfo {
        var totalMb = 0L
        var availMb = 0L
        try {
            File("/proc/meminfo").forEachLine { line ->
                if (line.startsWith("MemTotal:")) {
                    totalMb = parseKb(line) / 1024
                } else if (line.startsWith("MemAvailable:")) {
                    availMb = parseKb(line) / 1024
                }
            }
        } catch (_: Exception) {}
        return MemInfo(totalMb, availMb)
    }

    fun getBatteryPct(context: Context): Int {
        return try {
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
            bm?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
        } catch (_: Exception) {
            -1
        }
    }

    private fun parseKb(line: String): Long {
        val parts = line.split(Regex("\\s+"))
        return if (parts.size >= 2) parts[1].toLongOrNull() ?: 0L else 0L
    }

    data class MemInfo(val totalMb: Long, val availMb: Long) {
        val availGbStr: String get() = String.format("%.1f GB", availMb / 1024.0)
    }
}
