package utils

import java.io.{File, PrintWriter}
import scala.collection.mutable.ListBuffer
import scala.math.BigDecimal

case class LatencyStats(
  name: String,
  loadWeightsTime: Long = 0,
  setupTime: Long = 0,
  operationTime: Long = 0,
  pollingTime: Long = 0,
  totalTime: Long = 0
)

case class LatencyReportSummary(
  totalCyclesRaw: Long,
  totalTimeMsFromCycles: Double,
  totalTimeMsRoundedSum: Double,
  totalE2ETimeMsRoundedSum: Double
)

object LatencyTracker {
  private val stats = ListBuffer[LatencyStats]()

  private case class ReportRow(
    name: String,
    loadCyc: Long,
    setupCyc: Long,
    opCyc: Long,
    pollCyc: Long,
    totalCyc: Long,
    timeMsRaw: Double,
    timeMsRounded: Double,
    layerMultiplier: Double,
    e2eTimeMs: Double
  )

  def record(s: LatencyStats): Unit = stats += s

  def reset(): Unit = stats.clear()

  def getAllStats(): Seq[LatencyStats] = stats.toSeq

  def generateReport(
    filePath: String,
    clockFreqMhz: Double,
    simClockPeriod: Long,
    simulatedLayers: Int = 1,
    modelLayers: Int = 1,
    preLayerNames: Set[String] = Set.empty,
    postLayerNames: Set[String] = Set.empty
  ): LatencyReportSummary = {
    val safeSimLayers = if (simulatedLayers > 0) simulatedLayers else 1
    val layerMultiplier = modelLayers.toDouble / safeSimLayers.toDouble
    val rows = buildRows(clockFreqMhz, simClockPeriod, layerMultiplier, preLayerNames, postLayerNames)

    generateTxtReport(filePath, modelLayers, layerMultiplier, rows)

    val csvPath = if (filePath.endsWith(".txt")) {
      filePath.substring(0, filePath.length - 4) + ".csv"
    } else {
      filePath + ".csv"
    }
    generateCsvReport(csvPath, modelLayers, layerMultiplier, rows)

    val totalCyclesRaw = rows.map(_.totalCyc).sum
    val totalTimeMsFromCycles = round3(totalCyclesRaw / clockFreqMhz / 1000.0)
    val totalTimeMsRoundedSum = round3(rows.map(_.timeMsRounded).sum)
    val totalE2ETimeMsRoundedSum = round3(rows.map(_.e2eTimeMs).sum)

    LatencyReportSummary(
      totalCyclesRaw = totalCyclesRaw,
      totalTimeMsFromCycles = totalTimeMsFromCycles,
      totalTimeMsRoundedSum = totalTimeMsRoundedSum,
      totalE2ETimeMsRoundedSum = totalE2ETimeMsRoundedSum
    )
  }

  private def round3(value: Double): Double = {
    BigDecimal(value).setScale(3, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

  private def buildRows(
    clockFreqMhz: Double,
    simClockPeriod: Long,
    layerMultiplier: Double,
    preLayerNames: Set[String],
    postLayerNames: Set[String]
  ): Seq[ReportRow] = {
    def isInLayer(name: String): Boolean = !preLayerNames.contains(name) && !postLayerNames.contains(name)

    stats.toSeq.map { s =>
      val loadCyc = s.loadWeightsTime / simClockPeriod
      val setupCyc = s.setupTime / simClockPeriod
      val opCyc = s.operationTime / simClockPeriod
      val pollCyc = s.pollingTime / simClockPeriod
      val totalCyc = s.totalTime / simClockPeriod

      val timeMsRaw = totalCyc / clockFreqMhz / 1000.0
      val timeMsRounded = round3(timeMsRaw)
      val mul = if (isInLayer(s.name)) layerMultiplier else 1.0

      // E2E time is computed from per-module rounded ms and then scaled by layer multiplier.
      val e2eTimeMs = round3(timeMsRounded * mul)

      ReportRow(
        name = s.name,
        loadCyc = loadCyc,
        setupCyc = setupCyc,
        opCyc = opCyc,
        pollCyc = pollCyc,
        totalCyc = totalCyc,
        timeMsRaw = timeMsRaw,
        timeMsRounded = timeMsRounded,
        layerMultiplier = mul,
        e2eTimeMs = e2eTimeMs
      )
    }
  }

  private def generateTxtReport(
    filePath: String,
    modelLayers: Int,
    layerMultiplier: Double,
    rows: Seq[ReportRow]
  ): Unit = {
    val writer = new PrintWriter(new File(filePath))

    val totalCyclesRaw = rows.map(_.totalCyc).sum
    val totalMsRoundedSum = round3(rows.map(_.timeMsRounded).sum)
    val totalE2EMs = round3(rows.map(_.e2eTimeMs).sum)

    writer.println("=" * 190)
    writer.println("Latency report (224 x 224 Resolution)")
    writer.println(s"  Model layers: $modelLayers")
    writer.println("=" * 190)
    writer.println(f"${"Module Name"}%-30s | ${"Load (cyc)"}%12s | ${"Setup (cyc)"}%12s | ${"Op (cyc)"}%12s | ${"Poll (cyc)"}%12s | ${"Total (cyc)"}%12s | ${"Time (ms)"}%10s | ${"Layer Mult"}%10s | ${"E2E Time (ms)"}%14s")
    writer.println("-" * 190)

    rows.foreach { r =>
      writer.println(f"${r.name}%-30s | ${r.loadCyc}%12d | ${r.setupCyc}%12d | ${r.opCyc}%12d | ${r.pollCyc}%12d | ${r.totalCyc}%12d | ${r.timeMsRounded}%10.3f | ${r.layerMultiplier}%10.3f | ${r.e2eTimeMs}%14.3f")
    }

    writer.println("-" * 190)
    writer.println(f"${"TOTAL"}%-30s | ${""}%12s | ${""}%12s | ${""}%12s | ${""}%12s | ${totalCyclesRaw}%12d | ${totalMsRoundedSum}%10.3f | ${""}%10s | ${totalE2EMs}%14.3f")
    writer.println("=" * 190)
    writer.close()

    println(s"Latency TXT report generated at $filePath")
  }

  private def generateCsvReport(
    filePath: String,
    modelLayers: Int,
    layerMultiplier: Double,
    rows: Seq[ReportRow]
  ): Unit = {
    val writer = new PrintWriter(new File(filePath))

    val totalCyclesRaw = rows.map(_.totalCyc).sum
    val totalMsRoundedSum = round3(rows.map(_.timeMsRounded).sum)
    val totalE2EMs = round3(rows.map(_.e2eTimeMs).sum)

    writer.println(s"# model_layers=$modelLayers,in_layer_multiplier=$layerMultiplier")
    writer.println("Module Name,Load (cyc),Setup (cyc),Op (cyc),Poll (cyc),Total (cyc),Time (ms),Layer Multiplier,E2E Time (ms)")

    rows.foreach { r =>
      writer.println(f"${r.name},${r.loadCyc},${r.setupCyc},${r.opCyc},${r.pollCyc},${r.totalCyc},${r.timeMsRounded}%.3f,${r.layerMultiplier}%.3f,${r.e2eTimeMs}%.3f")
    }

    writer.println(f"TOTAL,,,,,${totalCyclesRaw},${totalMsRoundedSum}%.3f,,${totalE2EMs}%.3f")
    writer.close()

    println(s"Latency CSV report generated at $filePath")
  }
}
