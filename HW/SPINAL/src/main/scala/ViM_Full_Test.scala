import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axilite.AxiLite4
import spinal.core.sim._
import utils._

import scala.language.postfixOps

object simulate_vim_full extends App {
  
  val spinalConfig: SpinalConfig = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = SYNC, resetActiveLevel = LOW
    )
  )
  
  SimConfig
    .withConfig(spinalConfig)
    .withFstWave
    .withWaveDepth(1)
    .allOptimisation
    .withVerilator
    .addSimulatorFlag("--unroll-count 1024")
    .addSimulatorFlag("-j 16")
    .addSimulatorFlag("-O3 --x-assign fast --x-initial fast --noassert")
    .compile(new ViM_ACCELERATOR)
    .doSimUntilVoid { dut =>
      
      // Model parameters
      val D_MODEL = 192
      val D_INNER = 384
      val D_STATE = 16
      val NUM_PATCHES = 197
      val CLS_TOKEN_IDX = 98  // (224/16) * (224/16) / 2 = 14 * 14 / 2
      val CONV_KERNEL_SIZE = 4
      val DT_RANK = 16
      val NUM_LAYERS = 1
      val MODEL_NUM_LAYERS = 24
      var LAYER_IDX = 0
      val FEATURE_BLOCK_SIZE = 8
      val AXI_XFER_BIT_WIDTH_VAL = 256
      val CLOCK_FREQ_MHZ = 350.0
      val SIM_CLOCK_PERIOD = 10L

      val DEBUG_STOP_AFTER_IN_PROJ_X = false
      val opCyclesByName = scala.collection.mutable.Map[String, Long]()
      
      // Memory address layout (all in shared DRAM space)
      // Input/Output buffers
      val addr_input = 0x1000000L      // Initial input
      val addr_norm_out = 0x2000000L   // NORM output -> LINEAR(in_proj) input
      val addr_linear_inproj_out = 0x3000000L  // LINEAR(in_proj) output -> CONV input
      val addr_conv_out = 0x4000000L    // CONV output -> SMOOTH input
      val addr_smooth_out = 0x5000000L  // SMOOTH output -> LINEAR(x_proj_dt) input
      val addr_linear_xprojdt_out = 0x6000000L  // LINEAR(x_proj_dt) output -> LINEAR(dt_proj) input
      val addr_linear_dtproj_out = 0x7000000L   // LINEAR(dt_proj) output -> LINEAR(x_proj_bc_b) input
      val addr_linear_xprojbcb_out = 0x8000000L // LINEAR(x_proj_bc_b) output -> LINEAR(x_proj_bc_c) input
      val addr_state_c = 0x9000000L    // Final State C output
      val addr_linear_inprojz_out = 0xA000000L  // LINEAR(in_proj_z) output -> SSM input (z_silu)
      val addr_scan_out = 0xB000000L   // SSM output (out_z)
      val addr_outproj_smooth_in = 0x13000000L
      val addr_outproj_smooth_out = 0x14000000L
      val addr_outproj_out = 0x15000000L
      val addr_mixer_out = 0x16000000L
      
      // SSM input addresses
      val addr_ssm_u = 0xC000000L     // SSM input: u (CONV output)
      val addr_ssm_delta = 0xD000000L // SSM input: delta (dt_proj output)
      val addr_ssm_z_silu = 0xE000000L // SSM input: z_silu (in_proj_z output)
      val addr_ssm_B = 0xF000000L     // SSM input: B (x_proj_bc_b output)
      val addr_ssm_C = 0x10000000L    // SSM input: C (x_proj_bc_c output)
      
      // Weight addresses
      val addr_norm_weights = 0xA000000L
      val addr_linear_inproj_weights = 0xB000000L
      val addr_conv_weights = 0xC000000L
      val addr_smooth_weights = 0xD000000L
      val addr_linear_xprojdt_weights = 0xE000000L
      val addr_linear_dtproj_weights = 0xF000000L
      val addr_linear_xprojbcb_weights = 0x10000000L
      val addr_linear_xprojbcc_weights = 0x11000000L
      val addr_ssm_weights = 0x12000000L  // SSM weights (A and D buffers)
      val addr_linear_outproj_weights = 0x13000000L
      val addr_embed_weights = 0x14000000L
      val addr_embed_bias = 0x15000000L
      val addr_embed_pos_embed = 0x16000000L
      val addr_embed_cls_token = 0x17000000L
    
      // Data path
      val data_path_prefix = utils.DataPathConfig.data_path_prefix
      
      // SSM parameters
      val SCAN_DIM = D_INNER * D_STATE  // 384 * 16 = 6144
      
      println("=" * 80)
      println("=== ViM Full Pipeline Test ===")
      println("=" * 80)
      println(s"Clock Frequency: $CLOCK_FREQ_MHZ MHz")
      println(s"NUM_PATCHES: $NUM_PATCHES, D_MODEL: $D_MODEL, D_INNER: $D_INNER")
      println()
      
      // Track total cycles and latency
      var totalCycles: Long = 0
      var preLayerCycles: Long = 0
      var inLayerCycles: Long = 0
      var postLayerCycles: Long = 0
      val startTime = simTime()
      
      // Helper to track cycles
      val preLayerNames = Set("EMBED")
      val postLayerNames = Set("PATCH_OPS(load_cls)", "FINAL_NORM", "HEAD")
      def logModuleCycles(name: String, startTime: Long, endTime: Long): Unit = {
        val cycles = (endTime - startTime) / SIM_CLOCK_PERIOD
        totalCycles += cycles
        if (postLayerNames.contains(name)) postLayerCycles += cycles
        else if (preLayerNames.contains(name)) preLayerCycles += cycles
        else inLayerCycles += cycles
        val timeUs = cycles / CLOCK_FREQ_MHZ
        val timeMs = timeUs / 1000.0
        println(f"[$name] Cycles: $cycles | Time: $timeUs%.2f us ($timeMs%.3f ms)")
      }
      
      // Helper to write array to AXI memory
      def writeArrayToAxiMemory(
        axi_mem: AxiMemorySim,
        baseAddr: Long,
        data: Array[Long],
        numPatches: Int,
        dim: Int,
        blockSize: Int
      ): Unit = {
        val numBlocks = (dim + blockSize - 1) / blockSize
        for (p <- 0 until numPatches) {
          for (b <- 0 until numBlocks) {
            val block_idx = p * numBlocks + b
            val block_base_addr = (baseAddr + block_idx * blockSize * 4).toLong
            for (o <- 0 until blockSize) {
              val global_idx = p * dim + b * blockSize + o
              if (global_idx < data.length) {
                val value = BigInt(data(global_idx))
                val unsigned_value = if (value < 0) {
                  (BigInt(1) << 32) + value
                } else {
                  value
                }
                val elem_addr = block_base_addr + o * 4
                axi_mem.memory.writeBigInt(elem_addr, unsigned_value, 4)
              }
            }
          }
        }
      }
      
      // Helper to read array from AXI memory
      def readArrayFromAxiMemory(
        axi_mem: AxiMemorySim,
        baseAddr: Long,
        numPatches: Int,
        dim: Int,
        blockSize: Int
      ): Array[Long] = {
        val numBlocks = (dim + blockSize - 1) / blockSize
        val output = new Array[Long](numPatches * dim)
        for (p <- 0 until numPatches) {
          for (b <- 0 until numBlocks) {
            val block_idx = p * numBlocks + b
            val block_base_addr = (baseAddr + block_idx * blockSize * 4).toLong
            for (o <- 0 until blockSize) {
              val global_idx = p * dim + b * blockSize + o
              if (global_idx < output.length) {
                val elem_addr = block_base_addr + o * 4
                val unsigned_value = axi_mem.memory.readBigInt(elem_addr, 4)
                val signed_value = if (unsigned_value >= (BigInt(1) << 31)) {
                  unsigned_value - (BigInt(1) << 32)
                } else {
                  unsigned_value
                }
                output(global_idx) = signed_value.toLong
              }
            }
          }
        }
        output
      }
      
      // Helper to verify output
      def verifyOutput(
        dut_array: Array[Long],
        ref_file: String,
        numPatches: Int,
        dim: Int,
        test_name: String,
        fpType: FixedPointType = FixedPointTypes.fm_t,
        threshold: Double = 1.0,
        refData: Array[Long] = Array.empty
      ): Unit = {
        println(s"\n>>> Verifying $test_name...")
        if (ref_file.nonEmpty) println(s"  Reference file: $ref_file")
        println(s"  Expected dimensions: $numPatches patches × $dim features = ${numPatches * dim} samples")
        println(s"  DUT array length: ${dut_array.length}")
        
        // Check if reference file exists and read it
        val expected_samples = numPatches * dim
        
        val ref_float = if (ref_file.nonEmpty) {
          val floats = read_float_file(ref_file, expected_samples)
          println(s"  Reference file samples: ${floats.length}")
          floats
        } else if (refData.nonEmpty) {
          println(s"  Using provided reference data array (${refData.length} samples)")
          refData.map(l => FixedPointTypes.fixedToFloat(l, fpType))
        } else {
          println("  ERROR: No reference file or data provided!")
          return
        }
        
        if (ref_float.length != expected_samples) {
          println(s"  WARNING: Reference file has ${ref_float.length} samples, expected $expected_samples")
        }
        
        if (dut_array.length != expected_samples) {
          println(s"  ERROR: DUT array has ${dut_array.length} samples, expected $expected_samples")
          simFailure()
        }
        
        val ref_array = ref_float.map(f => 
          FixedPointTypes.floatToFixed(f, fpType).toLong
        )
        
        // Verify ALL samples - calculate MAE and MSE over complete dataset
        var sum_abs_error = 0.0
        var sum_sq_error = 0.0
        var max_abs_error = 0.0
        var valid_count = 0
        var mismatch_count = 0
        val worstK = 10
        val worstSamples = scala.collection.mutable.ArrayBuffer[(Double, Int, Double, Double, Double)]()
        def recordWorst(absErr: Double, idx: Int, dut: Double, ref: Double, err: Double): Unit = {
          if (worstSamples.length < worstK) {
            worstSamples.append((absErr, idx, dut, ref, err))
          } else {
            var minPos = 0
            var minVal = worstSamples(0)._1
            var j = 1
            while (j < worstSamples.length) {
              if (worstSamples(j)._1 < minVal) {
                minVal = worstSamples(j)._1
                minPos = j
              }
              j += 1
            }
            if (absErr > minVal) {
              worstSamples(minPos) = ((absErr, idx, dut, ref, err))
            }
          }
        }
        
        // Process ALL patches and ALL features
        println(s"  Processing all $expected_samples samples for MAE/MSE calculation...")
        for (i <- 0 until expected_samples) {
          if (i < dut_array.length && i < ref_float.length) {
            val dut_float = FixedPointTypes.fixedToFloat(dut_array(i), fpType)
            val ref_float_val = ref_float(i)
            val error = dut_float - ref_float_val
            val abs_error = scala.math.abs(error)
            
            sum_abs_error += abs_error
            sum_sq_error += error * error
            max_abs_error = scala.math.max(max_abs_error, abs_error)
            valid_count += 1
            recordWorst(abs_error, i, dut_float, ref_float_val, error)
            
            // Count significant mismatches (error > 1% of reference value, if reference is not near zero)
            if (scala.math.abs(ref_float_val) > 0.01 && abs_error > 0.01 * scala.math.abs(ref_float_val)) {
              mismatch_count += 1
            }
          } else {
            println(s"  WARNING: Sample index $i out of range (dut: ${dut_array.length}, ref: ${ref_float.length})")
          }
        }
        
        if (valid_count != expected_samples) {
          println(s"  ERROR: Only processed $valid_count samples, expected $expected_samples")
          simFailure()
        }
        
        val mae = sum_abs_error / valid_count
        val mse = sum_sq_error / valid_count
        val rmse = scala.math.sqrt(mse)
        
        println(f"\n  [VERIFICATION RESULTS - ALL DATA]")
        println(f"  Total samples processed: $valid_count/$expected_samples")
        println(f"  MAE (Mean Absolute Error): $mae%.6e (${if (mae < threshold) "✓ PASS" else "✗ FAIL"} - threshold: $threshold)")
        println(f"  MSE (Mean Squared Error): $mse%.6e (${if (mse < threshold) "✓ PASS" else "✗ FAIL"} - threshold: $threshold)")
        println(f"  RMSE (Root Mean Squared Error): $rmse%.6e")
        println(f"  Max Absolute Error: $max_abs_error%.6e")
        println(f"  Samples with >1%% relative error: $mismatch_count")
        
        val worstSorted = worstSamples.sortBy(-_._1)
        println(f"\n  [WORST ABS ERROR SAMPLES] (top ${worstSorted.length}):")
        for ((absErr, idx, dut, ref, err) <- worstSorted) {
          val rel_err = if (scala.math.abs(ref) > 1e-6) absErr / scala.math.abs(ref) * 100.0 else 0.0
          val patch = idx / dim
          val feat = idx % dim
          println(f"    idx=$idx (p=$patch f=$feat): DUT=$dut%.6f, REF=$ref%.6f, err=$err%.6e, abs_err=$absErr%.6e, rel_err=$rel_err%.2f%%")
        }
        
        // Show sample outputs at various positions
        val sample_indices = Seq(0, valid_count / 4, valid_count / 2, 3 * valid_count / 4, valid_count - 1).filter(_ < valid_count)
        println(f"\n  [SAMPLE COMPARISONS] (indices: ${sample_indices.mkString(", ")}):")
        for (idx <- sample_indices) {
          val dut_float = FixedPointTypes.fixedToFloat(dut_array(idx), fpType)
          val ref_float_val = ref_float(idx)
          val err = scala.math.abs(dut_float - ref_float_val)
          val rel_err = if (scala.math.abs(ref_float_val) > 1e-6) {
            err / scala.math.abs(ref_float_val) * 100.0
          } else {
            0.0
          }
          println(f"    [$idx] DUT=$dut_float%.6f, REF=$ref_float_val%.6f, abs_err=$err%.6e, rel_err=$rel_err%.2f%%")
        }
        
        // Check first and last patch to ensure data integrity
        println(f"\n  [DATA INTEGRITY CHECK]")
        println(f"  First patch (indices 0-${dim-1}):")
        for (i <- 0 until Math.min(8, dim)) {
          val dut_float = FixedPointTypes.fixedToFloat(dut_array(i), fpType)
          val ref_float_val = ref_float(i)
          println(f"    [$i] DUT=$dut_float%.6f, REF=$ref_float_val%.6f")
        }
        println(f"  Last patch (indices ${(numPatches-1)*dim}-${numPatches*dim-1}):")
        for (i <- (numPatches-1)*dim until Math.min(numPatches*dim, (numPatches-1)*dim + 8)) {
          val dut_float = FixedPointTypes.fixedToFloat(dut_array(i), fpType)
          val ref_float_val = ref_float(i)
          println(f"    [$i] DUT=$dut_float%.6f, REF=$ref_float_val%.6f")
        }

        if (mae >= threshold || mse >= threshold) {
          println(f"\n  ✗ VERIFICATION FAILED: MAE or MSE exceeds threshold of $threshold")
          println(f"     MAE: $mae >= $threshold: ${mae >= threshold}")
          println(f"     MSE: $mse >= $threshold: ${mse >= threshold}")
          simFailure()
        } else {
          println(f"\n  ✓ VERIFICATION PASSED: MAE and MSE both below threshold of $threshold")
        }
      }
      
      // Initialize AXI memory simulators (shared DRAM space)
      println("\n[SETUP] Initializing AXI memory simulators...")
      val axiMemCfg = AxiMemorySimConfig(
        maxOutstandingReads = 256,
        maxOutstandingWrites = 256,
        readResponseDelay = 0,
        writeResponseDelay = 0,
        interruptProbability = 0,
        interruptMaxDelay = 0
      )
      println(s"[SETUP] AXI memory sim config: $axiMemCfg")

      val embed_out_r_mem = AxiMemorySim(dut.io.m_axi_embed_out_r, dut.clockDomain, axiMemCfg)
      val embed_in_r_mem = AxiMemorySim(dut.io.m_axi_embed_in_r, dut.clockDomain, axiMemCfg)
      val embed_weights_mem = AxiMemorySim(dut.io.m_axi_embed_weights, dut.clockDomain, axiMemCfg)
      
      val norm_sum_in_a_mem = AxiMemorySim(dut.io.m_axi_norm_sum_in_a, dut.clockDomain, axiMemCfg)
      val norm_sum_in_b_mem = AxiMemorySim(dut.io.m_axi_norm_sum_in_b, dut.clockDomain, axiMemCfg)
      val norm_sum_out_r_mem = AxiMemorySim(dut.io.m_axi_norm_sum_out_r, dut.clockDomain, axiMemCfg)
      val norm_sum_weights_mem = AxiMemorySim(dut.io.m_axi_norm_sum_weights, dut.clockDomain, axiMemCfg)
      
      val linear_in_mem = AxiMemorySim(dut.io.m_axi_linear_in, dut.clockDomain, axiMemCfg)
      val linear_out_mem = AxiMemorySim(dut.io.m_axi_linear_out, dut.clockDomain, axiMemCfg)
      val linear_weights_mem = AxiMemorySim(dut.io.m_axi_linear_weights, dut.clockDomain, axiMemCfg)
      
      val conv_in_mem = AxiMemorySim(dut.io.m_axi_conv_in, dut.clockDomain, axiMemCfg)
      val conv_out_mem = AxiMemorySim(dut.io.m_axi_conv_out, dut.clockDomain, axiMemCfg)
      val conv_weights_mem = AxiMemorySim(dut.io.m_axi_conv_weights, dut.clockDomain, axiMemCfg)
      
      val smooth_in_mem = AxiMemorySim(dut.io.m_axi_smooth_in, dut.clockDomain, axiMemCfg)
      val smooth_out_mem = AxiMemorySim(dut.io.m_axi_smooth_out, dut.clockDomain, axiMemCfg)
      val smooth_weights_mem = AxiMemorySim(dut.io.m_axi_smooth_weights, dut.clockDomain, axiMemCfg)
      
      val ssm_in_u_mem = AxiMemorySim(dut.io.m_axi_ssm_in_u, dut.clockDomain, axiMemCfg)
      val ssm_in_delta_mem = AxiMemorySim(dut.io.m_axi_ssm_in_delta, dut.clockDomain, axiMemCfg)
      val ssm_in_z_silu_mem = AxiMemorySim(dut.io.m_axi_ssm_in_z_silu, dut.clockDomain, axiMemCfg)
      val ssm_in_B_mem = AxiMemorySim(dut.io.m_axi_ssm_in_B, dut.clockDomain, axiMemCfg)
      val ssm_in_C_mem = AxiMemorySim(dut.io.m_axi_ssm_in_C, dut.clockDomain, axiMemCfg)
      val ssm_weights_A_mem = AxiMemorySim(dut.io.m_axi_ssm_weights_A, dut.clockDomain, axiMemCfg)
      val ssm_weights_D_mem = AxiMemorySim(dut.io.m_axi_ssm_weights_D, dut.clockDomain, axiMemCfg)
      val ssm_out_r_mem = AxiMemorySim(dut.io.m_axi_ssm_out_r, dut.clockDomain, axiMemCfg)

      val patch_ops_in_mem = AxiMemorySim(dut.io.m_axi_patch_ops_in, dut.clockDomain, axiMemCfg)
      val patch_ops_out_mem = AxiMemorySim(dut.io.m_axi_patch_ops_out, dut.clockDomain, axiMemCfg)

      // Pre-allocate memory pages to avoid page faults
      val mems = Seq(
        embed_out_r_mem, embed_in_r_mem, embed_weights_mem,
        norm_sum_in_a_mem, norm_sum_in_b_mem, norm_sum_out_r_mem, norm_sum_weights_mem,
        linear_in_mem, linear_out_mem, linear_weights_mem,
        conv_in_mem, conv_out_mem, conv_weights_mem,
        smooth_in_mem, smooth_out_mem, smooth_weights_mem,
        ssm_in_u_mem, ssm_in_delta_mem, ssm_in_z_silu_mem, ssm_in_B_mem, ssm_in_C_mem,
        ssm_weights_A_mem, ssm_weights_D_mem, ssm_out_r_mem,
        patch_ops_in_mem, patch_ops_out_mem
      )

      val addrs = Seq(
        addr_input, addr_norm_out, addr_linear_inproj_out, addr_conv_out, addr_smooth_out,
        addr_linear_xprojdt_out, addr_linear_dtproj_out, addr_linear_xprojbcb_out,
        addr_state_c, addr_linear_inprojz_out, addr_scan_out,
        addr_outproj_smooth_in, addr_outproj_smooth_out, addr_outproj_out, addr_mixer_out,
        addr_ssm_u, addr_ssm_delta, addr_ssm_z_silu, addr_ssm_B, addr_ssm_C,
        addr_norm_weights, addr_linear_inproj_weights, addr_conv_weights, addr_smooth_weights,
        addr_linear_xprojdt_weights, addr_linear_dtproj_weights, addr_linear_xprojbcb_weights,
        addr_linear_xprojbcc_weights, addr_ssm_weights, addr_linear_outproj_weights,
        addr_embed_weights, addr_embed_bias, addr_embed_pos_embed, addr_embed_cls_token
      )

      println("[SETUP] Pre-allocating memory pages...")
      mems.foreach { mem =>
        addrs.foreach(addr => mem.memory.write(addr, 0.toByte))
      }

      final class Axi4BurstStats {
        var arReq: Long = 0L
        var arBeats: Long = 0L
        var rBeats: Long = 0L
        var rLast: Long = 0L
        var awReq: Long = 0L
        var awBeats: Long = 0L
        var wBeats: Long = 0L
        var wLast: Long = 0L
        var bResp: Long = 0L

        val arLenHist: scala.collection.mutable.HashMap[Int, Long] = new scala.collection.mutable.HashMap[Int, Long]()
        val awLenHist: scala.collection.mutable.HashMap[Int, Long] = new scala.collection.mutable.HashMap[Int, Long]()

        def reset(): Unit = {
          arReq = 0L
          arBeats = 0L
          rBeats = 0L
          rLast = 0L
          awReq = 0L
          awBeats = 0L
          wBeats = 0L
          wLast = 0L
          bResp = 0L
          arLenHist.clear()
          awLenHist.clear()
        }

        private def incHist(hist: scala.collection.mutable.HashMap[Int, Long], k: Int): Unit = {
          val prev = hist.getOrElse(k, 0L)
          hist.update(k, prev + 1L)
        }

        def onAr(len: Int): Unit = {
          arReq += 1L
          arBeats += (len.toLong + 1L)
          incHist(arLenHist, len)
        }

        def onR(last: Boolean): Unit = {
          rBeats += 1L
          if (last) rLast += 1L
        }

        def onAw(len: Int): Unit = {
          awReq += 1L
          awBeats += (len.toLong + 1L)
          incHist(awLenHist, len)
        }

        def onW(last: Boolean): Unit = {
          wBeats += 1L
          if (last) wLast += 1L
        }

        def onB(): Unit = bResp += 1L

        private def topLens(hist: scala.collection.mutable.HashMap[Int, Long], k: Int): String = {
          hist.toSeq.sortBy { case (_, c) => -c }.take(k).map { case (len, c) => s"${len}:${c}" }.mkString(", ")
        }

        def summaryString: String = {
          val avgReadBurst = if (arReq == 0L) 0.0 else arBeats.toDouble / arReq.toDouble
          val avgWriteBurst = if (awReq == 0L) 0.0 else awBeats.toDouble / awReq.toDouble
          val readTop = topLens(arLenHist, 6)
          val writeTop = topLens(awLenHist, 6)
          f"AR(req=$arReq beats=$arBeats avgBurst=$avgReadBurst%.2f last=$rLast rBeats=$rBeats topLen={$readTop}) | " +
            f"AW(req=$awReq beats=$awBeats avgBurst=$avgWriteBurst%.2f bResp=$bResp wBeats=$wBeats wLast=$wLast topLen={$writeTop})"
        }
      }

      final class Axi4Monitor(label: String, bus: Axi4, clockDomain: ClockDomain) {
        val stats = new Axi4BurstStats()
        @volatile var enabled: Boolean = false

        def start(): Unit = {
          stats.reset()
          enabled = true
        }

        def stop(): Unit = enabled = false

        fork {
          while (true) {
            clockDomain.waitSampling()
            if (enabled) {
              if (bus.ar.valid.toBoolean && bus.ar.ready.toBoolean) {
                stats.onAr(bus.ar.payload.len.toInt)
              }
              if (bus.r.valid.toBoolean && bus.r.ready.toBoolean) {
                stats.onR(bus.r.payload.last.toBoolean)
              }
              if (bus.aw.valid.toBoolean && bus.aw.ready.toBoolean) {
                stats.onAw(bus.aw.payload.len.toInt)
              }
              if (bus.w.valid.toBoolean && bus.w.ready.toBoolean) {
                stats.onW(bus.w.payload.last.toBoolean)
              }
              if (bus.b.valid.toBoolean && bus.b.ready.toBoolean) {
                stats.onB()
              }
            }
          }
        }

        def dump(prefix: String): Unit = {
          println(s"[$prefix][$label] ${stats.summaryString}")
        }
      }

      val linearInMon = new Axi4Monitor("m_axi_linear_in", dut.io.m_axi_linear_in, dut.clockDomain)
      val linearOutMon = new Axi4Monitor("m_axi_linear_out", dut.io.m_axi_linear_out, dut.clockDomain)
      val linearWeightsMon = new Axi4Monitor("m_axi_linear_weights", dut.io.m_axi_linear_weights, dut.clockDomain)
      
      // Initialize AXI-Lite drivers
      println("[SETUP] Initializing AXI-Lite drivers...")
      val embed_axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_embed_control, dut.clockDomain)
      val embed_axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_embed_control_r, dut.clockDomain)
      val norm_sum_axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_norm_sum_control, dut.clockDomain)
      val norm_sum_axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_norm_sum_control_r, dut.clockDomain)
      val linear_axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_linear_control, dut.clockDomain)
      val linear_axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_linear_control_r, dut.clockDomain)
      val conv_axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_conv_control, dut.clockDomain)
      val conv_axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_conv_control_r, dut.clockDomain)
      val smooth_axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_smooth_control, dut.clockDomain)
      val smooth_axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_smooth_control_r, dut.clockDomain)
      val ssm_axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_ssm_control, dut.clockDomain)
      val ssm_axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_ssm_control_r, dut.clockDomain)

      val patch_ops_axilite_driver = AddressSeparableAxiLite4Driver(dut.io.s_axi_patch_ops_control, dut.clockDomain)
      val patch_ops_axilite_driver_r = AddressSeparableAxiLite4Driver(dut.io.s_axi_patch_ops_control_r, dut.clockDomain)
      
      // Initialize daisychain
      init_daisy_chain(dut.io.signals)
      init_clock(dut.clockDomain, SIM_CLOCK_PERIOD.toInt)
      dut.clockDomain.waitSampling(10)
      
      // AXI-Lite register addresses
      val ADDR_AP_CTRL = 0x00
      val ADDR_OUT_DIM = 0x10
      val ADDR_IN_DIM = 0x18
      val ADDR_NUM_PATCHES = 0x18  // For NORM (address width 5 bits, max 0x1F)
      val ADDR_NUM_PATCHES_LINEAR = 0x20  // For LINEAR (address width 6 bits)
      val ADDR_FLAGS = 0x28
      val ADDR_INNER_DIM = 0x10  // For SMOOTH
      val ADDR_MODEL_DIM = 0x10  // For NORM
      val ADDR_CONV_DIM = 0x10   // For CONV
      
      // EMBED register offsets
      val ADDR_EMBED_OUT_DATA_0 = 0x10
      val ADDR_EMBED_IN_DATA_0 = 0x1c
      val ADDR_EMBED_WEIGHTS_DATA_0 = 0x28
      val ADDR_EMBED_BIAS_DATA_0 = 0x34
      val ADDR_EMBED_POS_EMBED_DATA_0 = 0x40
      val ADDR_EMBED_CLS_TOKEN_DATA_0 = 0x4c
      
      // EMBED scalar parameter addresses
      val ADDR_EMBED_DIM_DATA_0 = 0x10
      val ADDR_EMBED_NUMPATCHES_DATA_0 = 0x18
      val ADDR_EMBED_IMGHEIGHT_DATA_0 = 0x20
      val ADDR_EMBED_IMGWIDTH_DATA_0 = 0x28
      
      def resetModule(
        mems: Seq[AxiMemorySim],
        drivers: Seq[AddressSeparableAxiLite4Driver],
        waitCycles: Int = 100
      ): Unit = {
        mems.foreach(_.reset())
        drivers.foreach(_.reset())
        dut.clockDomain.waitSampling(waitCycles)
      }

      def writeAddr64(driver_r: AddressSeparableAxiLite4Driver, lowReg: Int, addr: Long): Unit = {
        driver_r.write(lowReg, (addr & 0xFFFFFFFFL).toInt)
        dut.clockDomain.waitSampling(5)
        driver_r.write(lowReg + 4, ((addr >>> 32) & 0xFFFFFFFFL).toInt)
        dut.clockDomain.waitSampling(5)
      }

      def writeBigIntToMem(mem: AxiMemorySim, addr: Long, data: BigInt, bytes: Int): Unit = {
        for (i <- 0 until bytes) {
          val byte = (data >> (i * 8)) & 0xFF
          mem.memory.write(addr + i, byte.toByte)
        }
      }

      def apStart(driver: AddressSeparableAxiLite4Driver, apCtrlReg: Int = ADDR_AP_CTRL): Unit = {
        driver.write(apCtrlReg, 0x11)
        dut.clockDomain.waitSampling()
        driver.write(apCtrlReg, 0x10)
      }

      def waitApDone(
        name: String,
        driver: AddressSeparableAxiLite4Driver,
        apCtrlReg: Int = ADDR_AP_CTRL,
        pollLimit: Int = 1000000,
        pollStep: Int = 10
      ): Long = {
        var done = false
        var pollCycles = 0
        while (!done && pollCycles < pollLimit) {
          val ctrl = driver.read(apCtrlReg)
          done = (ctrl & 0x02) != 0
          if (!done) {
            dut.clockDomain.waitSampling(pollStep)
            pollCycles += pollStep
          }
        }
        val endTime = simTime()
        if (!done) {
          println(s"ERROR: $name did not complete!")
          simFailure()
        }
        endTime
      }

      // NORM_SUM modes
      val NORM_SUM_BOTH = 0
      val NORM_SUM_NORM_ONLY = 1
      val NORM_SUM_ADD_ONLY = 2
      val NORM_SUM_DIV2_ONLY = 3

      def runNormSum(
        name: String,
        mode: Int,
        inputA: Array[Long],
        inputB: Array[Long],
        numPatches: Int,
        dim: Int,
        addrSrcA: Long,
        addrSrcB: Long,
        addrDst: Long,
        addrWeights: Long,
        weights: Array[Long],
        refFile: String = ""
      ): Array[Long] = {
        println(s"\n--- Running $name (Mode: $mode) ---")
        val tStart = simTime()
        
        // 1. Load weights (only needed for NORM modes)
        val tLoadStart = simTime()
        if (mode == NORM_SUM_BOTH || mode == NORM_SUM_NORM_ONLY) {
          val WEIGHTS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 16
          val weight_words = (dim + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD
          for (w <- 0 until weight_words) {
            val word_base_addr = (addrWeights + w * 32).toLong
            for (j <- 0 until WEIGHTS_PER_WORD) {
              val idx = w * WEIGHTS_PER_WORD + j
              if (idx < dim) {
                val value = BigInt(weights(idx))
                val unsigned_value = if (value < 0) (BigInt(1) << 16) + value else value
                norm_sum_weights_mem.memory.writeBigInt(word_base_addr + j * 2, unsigned_value, 2)
              }
            }
          }
        }
        val tLoadEnd = simTime()
        
        // 2. Setup
        val tSetupStart = simTime()
        resetModule(Seq(norm_sum_in_a_mem, norm_sum_in_b_mem, norm_sum_out_r_mem, norm_sum_weights_mem), Seq(norm_sum_axilite_driver, norm_sum_axilite_driver_r))
        
        // Write input A
        writeArrayToAxiMemory(norm_sum_in_a_mem, addrSrcA, inputA, numPatches, dim, FEATURE_BLOCK_SIZE)
        // Write input B (if needed)
        if (mode == NORM_SUM_BOTH || mode == NORM_SUM_ADD_ONLY || mode == NORM_SUM_DIV2_ONLY) {
          writeArrayToAxiMemory(norm_sum_in_b_mem, addrSrcB, inputB, numPatches, dim, FEATURE_BLOCK_SIZE)
        }
        
        // Configure addresses in control_r
        writeAddr64(norm_sum_axilite_driver_r, 0x10, addrDst)
        writeAddr64(norm_sum_axilite_driver_r, 0x1c, addrSrcA)
        writeAddr64(norm_sum_axilite_driver_r, 0x28, addrSrcB)
        writeAddr64(norm_sum_axilite_driver_r, 0x34, addrWeights)
        
        // Configure parameters in control
        norm_sum_axilite_driver.write(0x10, dim)         // model_dim
        norm_sum_axilite_driver.write(0x18, numPatches)  // num_patches
        norm_sum_axilite_driver.write(0x20, mode)         // mode
        
        dut.clockDomain.waitSampling(10)
        val tSetupEnd = simTime()
        
        // 3. Operation
        val tOpStart = simTime()
        apStart(norm_sum_axilite_driver)
        
        // 4. Polling
        val tOpEnd = waitApDone(name, norm_sum_axilite_driver)
        val tEnd = simTime()
        
        LatencyTracker.record(LatencyStats(
          name = name,
          loadWeightsTime = tLoadEnd - tLoadStart,
          setupTime = tSetupEnd - tSetupStart,
          operationTime = tOpEnd - tOpStart,
          pollingTime = tEnd - tOpEnd,
          totalTime = tEnd - tStart
        ))
        
        logModuleCycles(name, tOpStart, tOpEnd)
        val output = readArrayFromAxiMemory(norm_sum_out_r_mem, addrDst, numPatches, dim, FEATURE_BLOCK_SIZE)
        if (refFile.nonEmpty) verifyOutput(output, refFile, numPatches, dim, name)
        output
      }

      def runEmbed(
        name: String,
        numPatches: Int,
        dim: Int,
        addrSrc: Long,
        addrDst: Long,
        addrWeights: Long,
        addrBias: Long,
        addrPosEmbed: Long,
        addrClsToken: Long,
        imageData: Array[Float],
        weights: Array[Float],
        bias: Array[Float],
        posEmbed: Array[Float],
        clsToken: Array[Float],
        refFile: String = ""
      ): Array[Long] = {
        println(s"\n--- Running $name ---")
        val tStart = simTime()
        
        // 1. Load weights and other parameters
        val tLoadStart = simTime()
        
        // Load image data (assuming 32-bit floats for now, need to check HLS)
        for (i <- imageData.indices) {
          val fixed = FixedPointTypes.floatToFixed(imageData(i), FixedPointTypes.pixel_t)
          writeBigIntToMem(embed_in_r_mem, addrSrc + i * 4, fixed, 4)
        }
        
        // Load weights (16-bit wt_patch_embed_t)
        for (i <- weights.indices) {
          val fixed = FixedPointTypes.floatToFixed(weights(i), FixedPointTypes.wt_patch_embed_t)
          writeBigIntToMem(embed_weights_mem, addrWeights + i * 2, fixed, 2)
        }
        
        // Load bias (16-bit wt_patch_bias_t)
        for (i <- bias.indices) {
          val fixed = FixedPointTypes.floatToFixed(bias(i), FixedPointTypes.wt_patch_bias_t)
          writeBigIntToMem(embed_weights_mem, addrBias + i * 2, fixed, 2)
        }
        
        // Load pos_embed (32-bit fm_t)
        for (i <- posEmbed.indices) {
          val fixed = FixedPointTypes.floatToFixed(posEmbed(i), FixedPointTypes.fm_t)
          writeBigIntToMem(embed_weights_mem, addrPosEmbed + i * 4, fixed, 4)
        }
        
        // Load cls_token (32-bit fm_t)
        for (i <- clsToken.indices) {
          val fixed = FixedPointTypes.floatToFixed(clsToken(i), FixedPointTypes.fm_t)
          writeBigIntToMem(embed_weights_mem, addrClsToken + i * 4, fixed, 4)
        }
        
        val tLoadEnd = simTime()
        
        // 2. Setup
        val tSetupStart = simTime()
        resetModule(Seq(embed_in_r_mem, embed_out_r_mem, embed_weights_mem), Seq(embed_axilite_driver, embed_axilite_driver_r))
        
        // Configure addresses in control_r
        writeAddr64(embed_axilite_driver_r, ADDR_EMBED_OUT_DATA_0, addrDst)
        writeAddr64(embed_axilite_driver_r, ADDR_EMBED_IN_DATA_0, addrSrc)
        writeAddr64(embed_axilite_driver_r, ADDR_EMBED_WEIGHTS_DATA_0, addrWeights)
        writeAddr64(embed_axilite_driver_r, ADDR_EMBED_BIAS_DATA_0, addrBias)
        writeAddr64(embed_axilite_driver_r, ADDR_EMBED_POS_EMBED_DATA_0, addrPosEmbed)
        writeAddr64(embed_axilite_driver_r, ADDR_EMBED_CLS_TOKEN_DATA_0, addrClsToken)
        
        // Write scalar parameters via s_axi_control (not s_axi_control_r)
        embed_axilite_driver.write(ADDR_EMBED_DIM_DATA_0, dim)
        embed_axilite_driver.write(ADDR_EMBED_NUMPATCHES_DATA_0, numPatches)
        embed_axilite_driver.write(ADDR_EMBED_IMGHEIGHT_DATA_0, 224)  // INPUT_HEIGHT
        embed_axilite_driver.write(ADDR_EMBED_IMGWIDTH_DATA_0, 224)  // INPUT_WIDTH
        
        dut.clockDomain.waitSampling(10)
        val tSetupEnd = simTime()
        
        // 3. Operation
        val tOpStart = simTime()
        // EMBED is the first module in the DaisyChain
        dut.io.signals.I.T #= true
        dut.clockDomain.waitSampling()
        dut.io.signals.I.T #= false
        
        apStart(embed_axilite_driver)
        
        // 4. Polling
        val tOpEnd = waitApDone(name, embed_axilite_driver)
        val tEnd = simTime()
        
        LatencyTracker.record(LatencyStats(
          name = name,
          loadWeightsTime = tLoadEnd - tLoadStart,
          setupTime = tSetupEnd - tSetupStart,
          operationTime = tOpEnd - tOpStart,
          pollingTime = tEnd - tOpEnd,
          totalTime = tEnd - tStart
        ))
        
        logModuleCycles(name, tOpStart, tOpEnd)
        val output = readArrayFromAxiMemory(embed_out_r_mem, addrDst, numPatches, dim, FEATURE_BLOCK_SIZE)
        if (refFile.nonEmpty) verifyOutput(output, refFile, numPatches, dim, name)
        output
      }

      def runConv(
        name: String,
        inputData: Array[Long],
        numPatches: Int,
        dim: Int,
        addrSrc: Long,
        addrDst: Long,
        addrWeightsBase: Long,
        weightsMags: Array[Int],
        weightsSigns: Array[Int],
        bias: Array[Long],
        scales: Array[Long],
        refFile: String = ""
      ): Array[Long] = {
        println(s"\n--- Running $name ---")
        val tStart = simTime()
        
        // 1. Load weights/bias/scales
        val tLoadStart = simTime()
        write_int4_array_to_axi(conv_weights_mem, dut.clockDomain, addrWeightsBase, weightsMags, CONV_KERNEL_SIZE * dim, AXI_XFER_BIT_WIDTH_VAL, s"$name Mags")
        write_bit_array_to_axi(conv_weights_mem, dut.clockDomain, addrWeightsBase + 0x100000L, weightsSigns, CONV_KERNEL_SIZE * dim, AXI_XFER_BIT_WIDTH_VAL, s"$name Signs")
        
        val BIAS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
        val bias_words = (dim + BIAS_PER_WORD - 1) / BIAS_PER_WORD
        for (w <- 0 until bias_words) {
          val word_base_addr = (addrWeightsBase + 0x200000L + w * 32).toLong
          for (j <- 0 until BIAS_PER_WORD) {
            val idx = w * BIAS_PER_WORD + j
            if (idx < dim) {
              val value = BigInt(bias(idx))
              val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
              conv_weights_mem.memory.writeBigInt(word_base_addr + j * 4, unsigned_value, 4)
            }
          }
        }
        
        val SCALES_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
        val scale_words = (dim + SCALES_PER_WORD - 1) / SCALES_PER_WORD
        for (w <- 0 until scale_words) {
          val word_base_addr = (addrWeightsBase + 0x300000L + w * 32).toLong
          for (j <- 0 until SCALES_PER_WORD) {
            val idx = w * SCALES_PER_WORD + j
            if (idx < dim) {
              val value = BigInt(scales(idx))
              val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
              conv_weights_mem.memory.writeBigInt(word_base_addr + j * 4, unsigned_value, 4)
            }
          }
        }
        val tLoadEnd = simTime()
        
        // 2. Setup
        val tSetupStart = simTime()
        writeArrayToAxiMemory(conv_in_mem, addrSrc, inputData, numPatches, dim, FEATURE_BLOCK_SIZE)
        resetModule(Seq(conv_in_mem, conv_out_mem, conv_weights_mem), Seq(conv_axilite_driver, conv_axilite_driver_r))
        writeAddr64(conv_axilite_driver_r, 0x10, addrDst)
        writeAddr64(conv_axilite_driver_r, 0x1c, addrSrc)
        writeAddr64(conv_axilite_driver_r, 0x28, addrWeightsBase)
        writeAddr64(conv_axilite_driver_r, 0x34, addrWeightsBase + 0x100000L)
        writeAddr64(conv_axilite_driver_r, 0x40, addrWeightsBase + 0x200000L)
        writeAddr64(conv_axilite_driver_r, 0x4c, addrWeightsBase + 0x300000L)
        conv_axilite_driver.write(ADDR_CONV_DIM, dim)
        conv_axilite_driver.write(0x18, numPatches)  // num_patches
        dut.clockDomain.waitSampling(10)
        val tSetupEnd = simTime()
        
        // 3. Operation
        val tOpStart = simTime()
        apStart(conv_axilite_driver)
        
        // 4. Polling
        val tOpEnd = waitApDone(name, conv_axilite_driver)
        val tEnd = simTime()
        
        LatencyTracker.record(LatencyStats(
          name = name,
          loadWeightsTime = tLoadEnd - tLoadStart,
          setupTime = tSetupEnd - tSetupStart,
          operationTime = tOpEnd - tOpStart,
          pollingTime = tEnd - tOpEnd,
          totalTime = tEnd - tStart
        ))
        
        logModuleCycles(name, tOpStart, tOpEnd)
        val output = readArrayFromAxiMemory(conv_out_mem, addrDst, numPatches, dim, FEATURE_BLOCK_SIZE)
        if (refFile.nonEmpty) verifyOutput(output, refFile, numPatches, dim, name)
        output
      }

      def runSmooth(
        name: String,
        inputData: Array[Long],
        numPatches: Int,
        dim: Int,
        addrSrc: Long,
        addrDst: Long,
        addrWeights: Long,
        scales: Array[Long]
      ): Array[Long] = {
        println(s"\n--- Running $name ---")
        val tStart = simTime()
        
        // 1. Load weights
        val tLoadStart = simTime()
        val SCALES_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
        val scale_words = (dim + SCALES_PER_WORD - 1) / SCALES_PER_WORD
        for (w <- 0 until scale_words) {
          val word_base_addr = (addrWeights + w * 32).toLong
          for (j <- 0 until SCALES_PER_WORD) {
            val idx = w * SCALES_PER_WORD + j
            if (idx < dim) {
              val value = BigInt(scales(idx))
              val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
              smooth_weights_mem.memory.writeBigInt(word_base_addr + j * 4, unsigned_value, 4)
            }
          }
        }
        val tLoadEnd = simTime()
        
        // 2. Setup
        val tSetupStart = simTime()
        writeArrayToAxiMemory(smooth_in_mem, addrSrc, inputData, numPatches, dim, FEATURE_BLOCK_SIZE)
        resetModule(Seq(smooth_in_mem, smooth_out_mem, smooth_weights_mem), Seq(smooth_axilite_driver, smooth_axilite_driver_r))
        writeAddr64(smooth_axilite_driver_r, 0x10, addrDst)
        writeAddr64(smooth_axilite_driver_r, 0x1c, addrSrc)
        writeAddr64(smooth_axilite_driver_r, 0x28, addrWeights)
        smooth_axilite_driver.write(ADDR_INNER_DIM, dim)
        smooth_axilite_driver.write(0x18, numPatches)  // num_patches
        dut.clockDomain.waitSampling(10)
        val tSetupEnd = simTime()
        
        // 3. Operation
        val tOpStart = simTime()
        apStart(smooth_axilite_driver)
        
        // 4. Polling
        val tOpEnd = waitApDone(name, smooth_axilite_driver)
        val tEnd = simTime()
        
        LatencyTracker.record(LatencyStats(
          name = name,
          loadWeightsTime = tLoadEnd - tLoadStart,
          setupTime = tSetupEnd - tSetupStart,
          operationTime = tOpEnd - tOpStart,
          pollingTime = tEnd - tOpEnd,
          totalTime = tEnd - tStart
        ))
        
        logModuleCycles(name, tOpStart, tOpEnd)
        readArrayFromAxiMemory(smooth_out_mem, addrDst, numPatches, dim, FEATURE_BLOCK_SIZE)
      }

      def runSSM(
        name: String,
        u: Array[Long],
        delta: Array[Long],
        z_silu: Array[Long],
        B: Array[Long],
        C: Array[Long],
        numPatches: Int,
        dInner: Int,
        dState: Int,
        scanDim: Int,
        addrDst: Long,
        addrU: Long,
        addrDelta: Long,
        addrZSilu: Long,
        addrB: Long,
        addrC: Long,
        addrA: Long,
        addrD: Long,
        A_data: Array[Long],
        D_data: Array[Long],
        refFile: String = ""
      ): Array[Long] = {
        println(s"\n--- Running $name ---")
        val tStart = simTime()
        
        // 1. Load weights (A and D)
        val tLoadStart = simTime()
        val A_ELEMS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
        val A_words = (scanDim + A_ELEMS_PER_WORD - 1) / A_ELEMS_PER_WORD
        for (w <- 0 until A_words) {
          val word_base_addr = (addrA + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
          for (j <- 0 until A_ELEMS_PER_WORD) {
            val idx = w * A_ELEMS_PER_WORD + j
            if (idx < scanDim) {
              val value = BigInt(A_data(idx))
              val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
              val elem_addr = word_base_addr + j * 4
              ssm_weights_A_mem.memory.writeBigInt(elem_addr, unsigned_value, 4)
            }
          }
        }

        val D_ELEMS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
        val D_words = (dInner + D_ELEMS_PER_WORD - 1) / D_ELEMS_PER_WORD
        for (w <- 0 until D_words) {
          val word_base_addr = (addrD + w * (AXI_XFER_BIT_WIDTH_VAL / 8)).toLong
          for (j <- 0 until D_ELEMS_PER_WORD) {
            val idx = w * D_ELEMS_PER_WORD + j
            if (idx < dInner) {
              val value = BigInt(D_data(idx))
              val unsigned_value = if (value < 0) (BigInt(1) << 32) + value else value
              val elem_addr = word_base_addr + j * 4
              ssm_weights_D_mem.memory.writeBigInt(elem_addr, unsigned_value, 4)
            }
          }
        }
        val tLoadEnd = simTime()
        
        // 2. Setup
        val tSetupStart = simTime()
        // Write inputs to memory
        writeArrayToAxiMemory(ssm_in_u_mem, addrU, u, numPatches, dInner, FEATURE_BLOCK_SIZE)
        writeArrayToAxiMemory(ssm_in_delta_mem, addrDelta, delta, numPatches, dInner, FEATURE_BLOCK_SIZE)
        writeArrayToAxiMemory(ssm_in_z_silu_mem, addrZSilu, z_silu, numPatches, dInner, FEATURE_BLOCK_SIZE)
        writeArrayToAxiMemory(ssm_in_B_mem, addrB, B, numPatches, dState, FEATURE_BLOCK_SIZE)
        writeArrayToAxiMemory(ssm_in_C_mem, addrC, C, numPatches, dState, FEATURE_BLOCK_SIZE)

        resetModule(
          Seq(ssm_in_u_mem, ssm_in_delta_mem, ssm_in_z_silu_mem, ssm_in_B_mem, ssm_in_C_mem, ssm_weights_A_mem, ssm_weights_D_mem, ssm_out_r_mem),
          Seq(ssm_axilite_driver, ssm_axilite_driver_r)
        )
        
        // Memory addresses in s_axi_control_r
        writeAddr64(ssm_axilite_driver_r, 0x10, addrDst)
        ssm_axilite_driver_r.write(0x18, 1) // OUT_R_R_CTRL
        
        writeAddr64(ssm_axilite_driver_r, 0x1c, addrU)
        ssm_axilite_driver_r.write(0x24, 1) // U_SRC_CTRL
        
        writeAddr64(ssm_axilite_driver_r, 0x28, addrDelta)
        ssm_axilite_driver_r.write(0x30, 1) // DELTA_SRC_CTRL
        
        writeAddr64(ssm_axilite_driver_r, 0x34, addrZSilu)
        ssm_axilite_driver_r.write(0x3c, 1) // Z_SILU_SRC_CTRL
        
        writeAddr64(ssm_axilite_driver_r, 0x40, addrA)
        ssm_axilite_driver_r.write(0x48, 1) // A_BASE_CTRL
        
        writeAddr64(ssm_axilite_driver_r, 0x4c, addrB)
        ssm_axilite_driver_r.write(0x54, 1) // B_SRC_CTRL
        
        writeAddr64(ssm_axilite_driver_r, 0x58, addrC)
        ssm_axilite_driver_r.write(0x60, 1) // C_SRC_CTRL
        
        writeAddr64(ssm_axilite_driver_r, 0x64, addrD)
        ssm_axilite_driver_r.write(0x6c, 1) // D_BASE_CTRL
        
        // Parameters in s_axi_control
        val ADDR_SCAN_DIM = 0x10
        val ADDR_INNER_DIM = 0x18
        val ADDR_NUM_PATCHES_SSM = 0x20
        ssm_axilite_driver.write(ADDR_SCAN_DIM, scanDim)
        ssm_axilite_driver.write(ADDR_INNER_DIM, dInner)
        ssm_axilite_driver.write(ADDR_NUM_PATCHES_SSM, numPatches)
        
        dut.clockDomain.waitSampling(10)
        val tSetupEnd = simTime()
        
        // 3. Operation
        val tOpStart = simTime()
        apStart(ssm_axilite_driver)
        
        // 4. Polling
        val tOpEnd = waitApDone(name, ssm_axilite_driver)
        val tEnd = simTime()
        
        LatencyTracker.record(LatencyStats(
          name = name,
          loadWeightsTime = tLoadEnd - tLoadStart,
          setupTime = tSetupEnd - tSetupStart,
          operationTime = tOpEnd - tOpStart,
          pollingTime = tEnd - tOpEnd,
          totalTime = tEnd - tStart
        ))
        
        logModuleCycles(name, tOpStart, tOpEnd)
        val output = readArrayFromAxiMemory(ssm_out_r_mem, addrDst, numPatches, dInner, FEATURE_BLOCK_SIZE)
        if (refFile.nonEmpty) verifyOutput(output, refFile, numPatches, dInner, name)
        output
      }

      def runPatchOps(
        name: String,
        mode: Int,
        inputData: Array[Long],
        numPatches: Int,
        dim: Int,
        addrSrc: Long,
        addrDst: Long,
        refFile: String = "",
        refNumPatches: Int = -1,
        refData: Array[Long] = Array.empty
      ): Array[Long] = {
        println(s"\n--- Running $name ---")
        val tStart = simTime()
        
        // 1. Setup
        val tSetupStart = simTime()
        resetModule(Seq(patch_ops_in_mem, patch_ops_out_mem), Seq(patch_ops_axilite_driver, patch_ops_axilite_driver_r))
        writeArrayToAxiMemory(patch_ops_in_mem, addrSrc, inputData, numPatches, dim, FEATURE_BLOCK_SIZE)
        
        writeAddr64(patch_ops_axilite_driver_r, 0x10, addrDst)
        writeAddr64(patch_ops_axilite_driver_r, 0x1c, addrSrc)
        val ADDR_PATCH_OPS_MODE = 0x10
        val ADDR_PATCH_OPS_NUM_PATCHES = 0x18
        val ADDR_PATCH_OPS_CLS_TOKEN_IDX = 0x20
        val ADDR_PATCH_OPS_INNER_DIM = 0x28
        val ADDR_PATCH_OPS_MODEL_DIM = 0x30
        patch_ops_axilite_driver.write(ADDR_PATCH_OPS_MODE, mode)
        patch_ops_axilite_driver.write(ADDR_PATCH_OPS_NUM_PATCHES, numPatches)
        patch_ops_axilite_driver.write(ADDR_PATCH_OPS_CLS_TOKEN_IDX, CLS_TOKEN_IDX)
        patch_ops_axilite_driver.write(ADDR_PATCH_OPS_INNER_DIM, D_INNER)
        patch_ops_axilite_driver.write(ADDR_PATCH_OPS_MODEL_DIM, D_MODEL)
        val tSetupEnd = simTime()
        
        // 2. Operation
        val tOpStart = simTime()
        apStart(patch_ops_axilite_driver)
        
        // 3. Polling
        val tOpEnd = waitApDone(name, patch_ops_axilite_driver)
        val tEnd = simTime()
        
        LatencyTracker.record(LatencyStats(
          name = name,
          setupTime = tSetupEnd - tSetupStart,
          operationTime = tOpEnd - tOpStart,
          pollingTime = tEnd - tOpEnd,
          totalTime = tEnd - tStart
        ))
        
        logModuleCycles(name, tOpStart, tOpEnd)
        val finalRefNumPatches = if (refNumPatches > 0) refNumPatches else numPatches
        val output = readArrayFromAxiMemory(patch_ops_out_mem, addrDst, finalRefNumPatches, dim, FEATURE_BLOCK_SIZE)
        if (refFile.nonEmpty || refData.nonEmpty) verifyOutput(output, refFile, finalRefNumPatches, dim, name, refData = refData)
        output
      }

      def runLinear(
        name: String,
        inputData: Array[Long],
        numPatches: Int,
        inDim: Int,
        outDim: Int,
        flags: Int,
        addrSrc: Long,
        addrDst: Long,
        addrWeightsBase: Long,
        weightsInt4: Array[Int],
        biasOpt: Option[Array[Long]],
        scales: Array[Long],
        fpType: FixedPointType = FixedPointTypes.fm_t,
        refFile: String = ""
      ): Array[Long] = {
        println(s"\n--- Running $name ---")
        val tStart = simTime()
        
        // 1. Load weights/bias/scales
        val tLoadStart = simTime()
        
        val totalWeightElems = outDim * inDim
        write_int4_array_to_axi(
          linear_weights_mem,
          dut.clockDomain,
          addrWeightsBase,
          weightsInt4,
          totalWeightElems,
          AXI_XFER_BIT_WIDTH_VAL,
          s"$name Weights"
        )
        
        // Write bias if present (at +0x200000L)
        val addrBiasBase = addrWeightsBase + 0x200000L
        biasOpt.foreach { biasData =>
          val BIAS_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
          val biasWords = (outDim + BIAS_PER_WORD - 1) / BIAS_PER_WORD
          for (w <- 0 until biasWords) {
            val wordBaseAddr = (addrBiasBase + w * 32).toLong
            for (j <- 0 until BIAS_PER_WORD) {
              val idx = w * BIAS_PER_WORD + j
              if (idx < outDim) {
                val value = BigInt(biasData(idx))
                val unsignedValue = if (value < 0) (BigInt(1) << 32) + value else value
                linear_weights_mem.memory.writeBigInt(wordBaseAddr + j * 4, unsignedValue, 4)
              }
            }
          }
        }
        
        // Write scales (at +0x300000L)
        val addrScalesBase = addrWeightsBase + 0x300000L
        val scalesPerRow = (inDim + 31) / 32
        val totalScaleElems = outDim * scalesPerRow
        val SCALES_PER_WORD = AXI_XFER_BIT_WIDTH_VAL / 32
        val scaleWords = (totalScaleElems + SCALES_PER_WORD - 1) / SCALES_PER_WORD
        for (w <- 0 until scaleWords) {
          val wordBaseAddr = (addrScalesBase + w * 32).toLong
          for (j <- 0 until SCALES_PER_WORD) {
            val idx = w * SCALES_PER_WORD + j
            if (idx < totalScaleElems) {
              val scaleVal = if (scales.length == outDim) scales(idx / scalesPerRow) else scales(idx)
              val value = BigInt(scaleVal)
              val unsignedValue = if (value < 0) (BigInt(1) << 32) + value else value
              linear_weights_mem.memory.writeBigInt(wordBaseAddr + j * 4, unsignedValue, 4)
            }
          }
        }
        val tLoadEnd = simTime()
        
        // 2. Setup
        val tSetupStart = simTime()
        writeArrayToAxiMemory(linear_in_mem, addrSrc, inputData, numPatches, inDim, FEATURE_BLOCK_SIZE)

        resetModule(Seq(linear_in_mem, linear_out_mem, linear_weights_mem), Seq(linear_axilite_driver, linear_axilite_driver_r))
        
        // Pointer offsets in s_axi_control_r
        writeAddr64(linear_axilite_driver_r, 0x10, addrDst)
        writeAddr64(linear_axilite_driver_r, 0x1c, addrSrc)
        writeAddr64(linear_axilite_driver_r, 0x28, addrWeightsBase)  // Packed 4-bit weights
        writeAddr64(linear_axilite_driver_r, 0x34, addrScalesBase)   // Scales (was 0x40)
        writeAddr64(linear_axilite_driver_r, 0x40, addrBiasBase)     // Bias (was 0x34)
        
        // Scalars in s_axi_control
        linear_axilite_driver.write(ADDR_OUT_DIM, outDim)
        linear_axilite_driver.write(ADDR_IN_DIM, inDim)
        linear_axilite_driver.write(ADDR_NUM_PATCHES_LINEAR, numPatches)
        linear_axilite_driver.write(ADDR_FLAGS, flags)

        linear_axilite_driver.write(0x04, 1)
        linear_axilite_driver.write(0x08, 1)
        linear_axilite_driver.write(0x0c, 1)
        val tSetupEnd = simTime()

        // 3. Operation
        val dumpAxi = name == "LINEAR(in_proj_x)"
        if (dumpAxi) {
          linearInMon.start()
          linearOutMon.start()
          linearWeightsMon.start()
        }
        val tOpStart = simTime()
        apStart(linear_axilite_driver)

        // 4. Completion
        val tOpEnd = waitApDone(name, linear_axilite_driver)
        
        if (dumpAxi) {
          dut.clockDomain.waitSampling(50)
          linearInMon.stop()
          linearOutMon.stop()
          linearWeightsMon.stop()
          linearInMon.dump(name)
          linearOutMon.dump(name)
          linearWeightsMon.dump(name)
        }
        val tEnd = simTime()
        opCyclesByName(name) = (tOpEnd - tOpStart) / SIM_CLOCK_PERIOD
        
        LatencyTracker.record(LatencyStats(
          name = name,
          loadWeightsTime = tLoadEnd - tLoadStart,
          setupTime = tSetupEnd - tSetupStart,
          operationTime = tOpEnd - tOpStart,
          pollingTime = tEnd - tOpEnd,
          totalTime = tEnd - tStart
        ))
        
        logModuleCycles(name, tOpStart, tOpEnd)
        
        val output = readArrayFromAxiMemory(linear_out_mem, addrDst, numPatches, outDim, FEATURE_BLOCK_SIZE)
        if (refFile.nonEmpty) {
          verifyOutput(output, refFile, numPatches, outDim, name, fpType)
        }
        output
      }
      
      def banner(title: String): Unit = {
        println("\n" + "=" * 80)
        println(title)
        println("=" * 80)
      }

      object Files {
        def imageFloat32(file: String): String = s"$data_path_prefix/image_float32_block/$file"
        def binFloat32(file: String): String = s"$data_path_prefix/bin_float32_block/$file"
        def refFloat32(file: String): String = s"$data_path_prefix/ref_float32_block/$file"

        def layerBin(layer: Int, file: String): String = binFloat32(s"layers.$layer.$file")
        def layerRef(layer: Int, file: String): String = refFloat32(s"layers.$layer.$file")
      }

      def floatArrayToFixed(values: Array[Float], fpType: FixedPointType): Array[Long] =
        values.map(f => FixedPointTypes.floatToFixed(f, fpType).toLong)

      def readFloatToFixed(file: String, count: Int, fpType: FixedPointType): Array[Long] =
        floatArrayToFixed(read_float_file(file, count), fpType)

      def expectNonEmpty(data: Array[Long], label: String): Array[Long] = {
        if (data.isEmpty) {
          println(s"ERROR: $label is empty")
          simFailure()
        }
        data
      }

      def reversePatches(in: Array[Long], numPatches: Int, dim: Int): Array[Long] = {
        val out = new Array[Long](numPatches * dim)
        for (p <- 0 until numPatches) {
          val srcP = numPatches - 1 - p
          val srcBase = srcP * dim
          val dstBase = p * dim
          for (d <- 0 until dim) out(dstBase + d) = in(srcBase + d)
        }
        out
      }

      val PATCH_OP_FLIP = 0

      var layerInput = Array.empty[Long]
      var lastLayerOutput = Array.empty[Long]

      banner("STAGE: EMBED")

      val embedImageFloat = read_float_file(Files.imageFloat32("image.float32.bin"), 3 * 224 * 224)
      val embedWeightFloat = read_float_file(Files.binFloat32("patch_embed.proj.weight.float32.bin"), D_MODEL * 3 * 16 * 16)
      val embedBiasFloat = read_float_file(Files.binFloat32("patch_embed.proj.bias.float32.bin"), D_MODEL)
      val embedPosEmbedFloat = read_float_file(Files.binFloat32("pos_embed.float32.bin"), NUM_PATCHES * D_MODEL)
      val embedClsTokenFloat = read_float_file(Files.binFloat32("cls_token.float32.bin"), D_MODEL)

      val embedOutput = runEmbed(
        name = "EMBED",
        numPatches = NUM_PATCHES,
        dim = D_MODEL,
        addrSrc = 0x20000000L,
        addrDst = addr_input,
        addrWeights = addr_embed_weights,
        addrBias = addr_embed_bias,
        addrPosEmbed = addr_embed_pos_embed,
        addrClsToken = addr_embed_cls_token,
        imageData = embedImageFloat,
        weights = embedWeightFloat,
        bias = embedBiasFloat,
        posEmbed = embedPosEmbedFloat,
        clsToken = embedClsTokenFloat,
        refFile = Files.refFloat32("layers.0.norm.input.float32.bin")
      )

      layerInput = embedOutput

      for (layerIdx <- 0 until NUM_LAYERS) {
        LAYER_IDX = layerIdx

        banner(s"LAYER $layerIdx")

        val normInput = expectNonEmpty(layerInput, "layerInput")

        banner("LAYER: NORM")

        val normWeightsFixed = readFloatToFixed(Files.layerBin(layerIdx, "norm.weight.float32.bin"), D_MODEL, FixedPointTypes.wt_norm_t)

        val normOutput = runNormSum(
          name = "NORM",
          mode = NORM_SUM_NORM_ONLY,
          inputA = normInput,
          inputB = Array.empty,
          numPatches = NUM_PATCHES,
          dim = D_MODEL,
          addrSrcA = addr_input,
          addrSrcB = 0,
          addrDst = addr_norm_out,
          addrWeights = addr_norm_weights,
          weights = normWeightsFixed,
          refFile = Files.layerRef(layerIdx, "mixer.in_proj.input.float32.bin")
        )

        banner("LAYER: LINEAR(in_proj)")

        val inProjTotalWeightElems = D_INNER * D_MODEL
        val inProjXWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.in_proj.weight.x.int4.bin"), inProjTotalWeightElems)
        val inProjZWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.in_proj.weight.z.int4.bin"), inProjTotalWeightElems)
        val inProjScalesPerRow = (D_MODEL + 31) / 32
        val inProjXScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.in_proj.weight_scale.x.float32.bin"),
          D_INNER * inProjScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )
        val inProjZScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.in_proj.weight_scale.z.float32.bin"),
          D_INNER * inProjScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )

        val inProjXOutput = runLinear(
          name = "LINEAR(in_proj_x)",
          inputData = normOutput,
          numPatches = NUM_PATCHES,
          inDim = D_MODEL,
          outDim = D_INNER,
          flags = 0,
          addrSrc = addr_norm_out,
          addrDst = addr_linear_inproj_out,
          addrWeightsBase = addr_linear_inproj_weights,
          weightsInt4 = inProjXWeightsInt4,
          biasOpt = Some(Array.fill[Long](D_INNER)(0L)),
          scales = inProjXScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.in_proj.output.x.float32.bin")
        )

        if (DEBUG_STOP_AFTER_IN_PROJ_X) {
          val cyc = opCyclesByName.getOrElse("LINEAR(in_proj_x)", -1L)
          val timeUs = cyc / CLOCK_FREQ_MHZ
          val timeMs = timeUs / 1000.0
          println(f"[DEBUG] LINEAR(in_proj_x) op cycles: $cyc | Time: $timeUs%.2f us ($timeMs%.3f ms)")
          dut.clockDomain.waitSampling(50)
          simSuccess()
        }

        val addrLinearInProjZWeights = addr_linear_inproj_weights + 0x500000L
        val inProjZSiLUOutput = runLinear(
          name = "LINEAR(in_proj_z)",
          inputData = normOutput,
          numPatches = NUM_PATCHES,
          inDim = D_MODEL,
          outDim = D_INNER,
          flags = 2,
          addrSrc = addr_norm_out,
          addrDst = addr_linear_inprojz_out,
          addrWeightsBase = addrLinearInProjZWeights,
          weightsInt4 = inProjZWeightsInt4,
          biasOpt = Some(Array.fill[Long](D_INNER)(0L)),
          scales = inProjZScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.in_proj.output.z_silu.float32.bin")
        )

        banner("LAYER: CONV")

        val convTotalWeightElems = CONV_KERNEL_SIZE * D_INNER
        val convWeightsMags = read_int4_file(Files.layerBin(layerIdx, "mixer.conv1d.weight.magnitude.bin"), convTotalWeightElems)
        val convWeightsSigns = read_bit_file(Files.layerBin(layerIdx, "mixer.conv1d.weight.sign.bin"), convTotalWeightElems)
        val convBiasFixed = readFloatToFixed(Files.layerBin(layerIdx, "mixer.conv1d.bias.float32.bin"), D_INNER, FixedPointTypes.wt_conv_bias_t)
        val convScalesFixed = readFloatToFixed(Files.layerBin(layerIdx, "mixer.conv1d.weight_scale.float32.bin"), D_INNER, FixedPointTypes.wt_conv_ws_t)

        val convOutput = runConv(
          name = "CONV",
          inputData = inProjXOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addr_linear_inproj_out,
          addrDst = addr_conv_out,
          addrWeightsBase = addr_conv_weights,
          weightsMags = convWeightsMags,
          weightsSigns = convWeightsSigns,
          bias = convBiasFixed,
          scales = convScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.x_proj.input.float32.bin")
        )

        banner("LAYER: SMOOTH")

        val smoothScalesFloat = read_float_file(Files.layerBin(layerIdx, "mixer.x_proj.smooth_scales.float32.bin"), D_INNER)
        val smoothScalesFixed = floatArrayToFixed(smoothScalesFloat, FixedPointTypes.wt_linear_ss_t)

        val smoothOutput = runSmooth(
          name = "SMOOTH",
          inputData = convOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addr_conv_out,
          addrDst = addr_smooth_out,
          addrWeights = addr_smooth_weights,
          scales = smoothScalesFixed
        )

        val smoothRefFloat = new Array[Float](NUM_PATCHES * D_INNER)
        for (p <- 0 until NUM_PATCHES) {
          for (d <- 0 until D_INNER) {
            val idx = p * D_INNER + d
            val convFloat = FixedPointTypes.fixedToFloat(convOutput(idx), FixedPointTypes.fm_t)
            smoothRefFloat(idx) = convFloat * smoothScalesFloat(d)
          }
        }

        var smoothSumAbsError = 0.0
        var smoothSumSqError = 0.0
        var smoothMaxAbsError = 0.0
        for (i <- 0 until smoothOutput.length) {
          val dutFloat = FixedPointTypes.fixedToFloat(smoothOutput(i), FixedPointTypes.fm_t)
          val refFloatVal = smoothRefFloat(i)
          val error = dutFloat - refFloatVal
          val absError = scala.math.abs(error)
          smoothSumAbsError += absError
          smoothSumSqError += error * error
          smoothMaxAbsError = scala.math.max(smoothMaxAbsError, absError)
        }
        val smoothMae = smoothSumAbsError / smoothOutput.length
        val smoothMse = smoothSumSqError / smoothOutput.length
        println(f"[SMOOTH] Verification: MAE=$smoothMae%.6e, MSE=$smoothMse%.6e, Max Error=$smoothMaxAbsError%.6e")

        banner("LAYER: LINEAR(x_proj_dt)")

        val xProjDtTotalWeightElems = DT_RANK * D_INNER
        val xProjDtWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.x_proj.weight.dt.int4.bin"), xProjDtTotalWeightElems)
        val xProjDtScalesPerRow = (D_INNER + 31) / 32
        val xProjDtScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.x_proj.weight_scale.dt.float32.bin"),
          DT_RANK * xProjDtScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )

        val xProjDtOutput = runLinear(
          name = "LINEAR(x_proj_dt)",
          inputData = smoothOutput,
          numPatches = NUM_PATCHES,
          inDim = D_INNER,
          outDim = DT_RANK,
          flags = 0,
          addrSrc = addr_smooth_out,
          addrDst = addr_linear_xprojdt_out,
          addrWeightsBase = addr_linear_xprojdt_weights,
          weightsInt4 = xProjDtWeightsInt4,
          biasOpt = None,
          scales = xProjDtScalesFixed
        )

        banner("LAYER: LINEAR(dt_proj)")

        val dtProjTotalWeightElems = D_INNER * DT_RANK
        val dtProjWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.dt_proj.weight.int4.bin"), dtProjTotalWeightElems)
        val dtProjBiasFixed = readFloatToFixed(Files.layerBin(layerIdx, "mixer.dt_proj.bias.float32.bin"), D_INNER, FixedPointTypes.wt_linear_bias_t)
        val dtProjScalesFixed = readFloatToFixed(Files.layerBin(layerIdx, "mixer.dt_proj.weight_scale.float32.bin"), D_INNER, FixedPointTypes.wt_linear_ws_t)

        val deltaOutput = runLinear(
          name = "LINEAR(dt_proj)",
          inputData = xProjDtOutput,
          numPatches = NUM_PATCHES,
          inDim = DT_RANK,
          outDim = D_INNER,
          flags = 5,
          addrSrc = addr_linear_xprojdt_out,
          addrDst = addr_linear_dtproj_out,
          addrWeightsBase = addr_linear_dtproj_weights,
          weightsInt4 = dtProjWeightsInt4,
          biasOpt = Some(dtProjBiasFixed),
          scales = dtProjScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.delta.float32.bin")
        )

        banner("LAYER: LINEAR(x_proj_bc)")

        val xProjBcTotalWeightElems = D_STATE * D_INNER
        val xProjBcScalesPerRow = (D_INNER + 31) / 32
        
        // Load B weights and scales
        val xProjBcBWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.x_proj.weight.b.int4.bin"), xProjBcTotalWeightElems)
        val xProjBcBScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.x_proj.weight_scale.b.float32.bin"),
          D_STATE * xProjBcScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )

        val bOutput = runLinear(
          name = "LINEAR(x_proj_bc_b)",
          inputData = smoothOutput,
          numPatches = NUM_PATCHES,
          inDim = D_INNER,
          outDim = D_STATE,
          flags = 0,
          addrSrc = addr_smooth_out,
          addrDst = addr_linear_xprojbcb_out,
          addrWeightsBase = addr_linear_xprojbcb_weights,
          weightsInt4 = xProjBcBWeightsInt4,
          biasOpt = None,
          scales = xProjBcBScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.B.float32.bin")
        )

        // Load C weights and scales
        val xProjBcCWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.x_proj.weight.c.int4.bin"), xProjBcTotalWeightElems)
        val xProjBcCScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.x_proj.weight_scale.c.float32.bin"),
          D_STATE * xProjBcScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )

        val cOutput = runLinear(
          name = "LINEAR(x_proj_bc_c)",
          inputData = smoothOutput,
          numPatches = NUM_PATCHES,
          inDim = D_INNER,
          outDim = D_STATE,
          flags = 0,
          addrSrc = addr_smooth_out,
          addrDst = addr_state_c,
          addrWeightsBase = addr_linear_xprojbcc_weights,
          weightsInt4 = xProjBcCWeightsInt4,
          biasOpt = None,
          scales = xProjBcCScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.C.float32.bin")
        )

        banner("LAYER: SSM")

        val aData = readFloatToFixed(Files.layerBin(layerIdx, "mixer.A.float32.bin"), SCAN_DIM, FixedPointTypes.fm_t)
        val dData = readFloatToFixed(Files.layerBin(layerIdx, "mixer.D.float32.bin"), D_INNER, FixedPointTypes.fm_t)

        val outZOutput = runSSM(
          name = "SSM",
          u = convOutput,
          delta = deltaOutput,
          z_silu = inProjZSiLUOutput,
          B = bOutput,
          C = cOutput,
          numPatches = NUM_PATCHES,
          dInner = D_INNER,
          dState = D_STATE,
          scanDim = SCAN_DIM,
          addrDst = addr_scan_out,
          addrU = addr_ssm_u,
          addrDelta = addr_ssm_delta,
          addrZSilu = addr_ssm_z_silu,
          addrB = addr_ssm_B,
          addrC = addr_ssm_C,
          addrA = addr_ssm_weights,
          addrD = addr_ssm_weights + 0x100000L,
          A_data = aData,
          D_data = dData,
          refFile = Files.layerRef(layerIdx, "mixer.out_z.float32.bin")
        )

        banner("LAYER: BACKWARD")

        val addrFlipXIn = 0x00100000L
        val addrFlipXOut = 0x00200000L
        val addrFlipZIn = 0x00300000L
        val addrFlipZOut = 0x00400000L
        val addrConvBOut = 0x00500000L
        val addrSmoothBOut = 0x00600000L
        val addrLinearXProjDtBOut = 0x00700000L
        val addrLinearDtProjBOut = 0x00800000L
        val addrLinearXProjBcBBOut = 0x00900000L
        val addrLinearXProjBcCBOut = 0x00A00000L
        val addrScanBOut = 0x00B00000L
        val addrSumDivideOut = 0x00C00000L

        val addrConvBWeights = 0x00100000L
        val addrSmoothBWeights = 0x00200000L
        val addrLinearXProjDtBWeights = 0x00300000L
        val addrLinearDtProjBWeights = 0x00400000L
        val addrLinearXProjBcBBWeights = 0x00500000L
        val addrLinearXProjBcCBWeights = 0x00600000L
        val addrScanBWeights = 0x00700000L

        val flipXRef = reversePatches(inProjXOutput, NUM_PATCHES, D_INNER)
        val flipXOutput = runPatchOps(
          name = "PATCH_OPS (flip x)",
          mode = PATCH_OP_FLIP,
          inputData = inProjXOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addrFlipXIn,
          addrDst = addrFlipXOut,
          refData = flipXRef
        )

        val flipZRef = reversePatches(inProjZSiLUOutput, NUM_PATCHES, D_INNER)
        val flipZOutput = runPatchOps(
          name = "PATCH_OPS (flip z_silu)",
          mode = PATCH_OP_FLIP,
          inputData = inProjZSiLUOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addrFlipZIn,
          addrDst = addrFlipZOut,
          refData = flipZRef
        )

        val convBOutput = runConv(
          name = "CONV_B",
          inputData = flipXOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addrFlipXOut,
          addrDst = addrConvBOut,
          addrWeightsBase = addrConvBWeights,
          weightsMags = read_int4_file(Files.layerBin(layerIdx, "mixer.conv1d_b.weight.magnitude.bin"), CONV_KERNEL_SIZE * D_INNER),
          weightsSigns = read_bit_file(Files.layerBin(layerIdx, "mixer.conv1d_b.weight.sign.bin"), CONV_KERNEL_SIZE * D_INNER),
          bias = readFloatToFixed(Files.layerBin(layerIdx, "mixer.conv1d_b.bias.float32.bin"), D_INNER, FixedPointTypes.wt_conv_bias_t),
          scales = readFloatToFixed(Files.layerBin(layerIdx, "mixer.conv1d_b.weight_scale.float32.bin"), D_INNER, FixedPointTypes.wt_conv_ws_t),
          refFile = Files.layerRef(layerIdx, "mixer.x_proj_b.input.float32.bin")
        )

        val smoothBScalesFixed = readFloatToFixed(Files.layerBin(layerIdx, "mixer.x_proj_b.smooth_scales.float32.bin"), D_INNER, FixedPointTypes.wt_linear_ss_t)
        val smoothBOutput = runSmooth(
          name = "SMOOTH_B",
          inputData = convBOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addrConvBOut,
          addrDst = addrSmoothBOut,
          addrWeights = addrSmoothBWeights,
          scales = smoothBScalesFixed
        )

        val xProjDtBTotalWeightElems = DT_RANK * D_INNER
        val xProjDtBWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.x_proj_b.weight.dt.int4.bin"), xProjDtBTotalWeightElems)
        val xProjDtBScalesPerRow = (D_INNER + 31) / 32
        val xProjDtBScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.x_proj_b.weight_scale.dt.float32.bin"),
          DT_RANK * xProjDtBScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )

        val xProjDtBOutput = runLinear(
          name = "LINEAR(x_proj_dt_b)",
          inputData = smoothBOutput,
          numPatches = NUM_PATCHES,
          inDim = D_INNER,
          outDim = DT_RANK,
          flags = 0,
          addrSrc = addrSmoothBOut,
          addrDst = addrLinearXProjDtBOut,
          addrWeightsBase = addrLinearXProjDtBWeights,
          weightsInt4 = xProjDtBWeightsInt4,
          biasOpt = None,
          scales = xProjDtBScalesFixed,
          refFile = ""
        )

        val dtProjBTotalWeightElems = D_INNER * DT_RANK
        val dtProjBWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.dt_proj_b.weight.int4.bin"), dtProjBTotalWeightElems)
        val dtProjBBiasFixed = readFloatToFixed(Files.layerBin(layerIdx, "mixer.dt_proj_b.bias.float32.bin"), D_INNER, FixedPointTypes.wt_linear_bias_t)
        val dtProjBScalesFixed = readFloatToFixed(Files.layerBin(layerIdx, "mixer.dt_proj_b.weight_scale.float32.bin"), D_INNER, FixedPointTypes.wt_linear_ws_t)

        val dtProjBOutput = runLinear(
          name = "LINEAR(dt_proj_b)",
          inputData = xProjDtBOutput,
          numPatches = NUM_PATCHES,
          inDim = DT_RANK,
          outDim = D_INNER,
          flags = 5,
          addrSrc = addrLinearXProjDtBOut,
          addrDst = addrLinearDtProjBOut,
          addrWeightsBase = addrLinearDtProjBWeights,
          weightsInt4 = dtProjBWeightsInt4,
          biasOpt = Some(dtProjBBiasFixed),
          scales = dtProjBScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.delta_b.float32.bin")
        )

        val xProjBcBTotalWeightElems = D_STATE * D_INNER
        val xProjBcBScalesPerRow = (D_INNER + 31) / 32
        
        // Load B_b weights and scales
        val xProjBcBBWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.x_proj_b.weight.b.int4.bin"), xProjBcBTotalWeightElems)
        val xProjBcBBScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.x_proj_b.weight_scale.b.float32.bin"),
          D_STATE * xProjBcBScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )
        
        // Load C_b weights and scales
        val xProjBcCBWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.x_proj_b.weight.c.int4.bin"), xProjBcBTotalWeightElems)
        val xProjBcCBScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.x_proj_b.weight_scale.c.float32.bin"),
          D_STATE * xProjBcBScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )

        val bBOutput = runLinear(
          name = "LINEAR(x_proj_bc_b)",
          inputData = smoothBOutput,
          numPatches = NUM_PATCHES,
          inDim = D_INNER,
          outDim = D_STATE,
          flags = 0,
          addrSrc = addrSmoothBOut,
          addrDst = addrLinearXProjBcBBOut,
          addrWeightsBase = addrLinearXProjBcBBWeights,
          weightsInt4 = xProjBcBBWeightsInt4,
          biasOpt = None,
          scales = xProjBcBBScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.B_b.float32.bin")
        )

        val cBOutput = runLinear(
          name = "LINEAR(x_proj_bc_c)",
          inputData = smoothBOutput,
          numPatches = NUM_PATCHES,
          inDim = D_INNER,
          outDim = D_STATE,
          flags = 0,
          addrSrc = addrSmoothBOut,
          addrDst = addrLinearXProjBcCBOut,
          addrWeightsBase = addrLinearXProjBcCBWeights,
          weightsInt4 = xProjBcCBWeightsInt4,
          biasOpt = None,
          scales = xProjBcCBScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.C_b.float32.bin")
        )

        val aBData = readFloatToFixed(Files.layerBin(layerIdx, "mixer.A_b.float32.bin"), SCAN_DIM, FixedPointTypes.fm_t)
        val dBData = readFloatToFixed(Files.layerBin(layerIdx, "mixer.D_b.float32.bin"), D_INNER, FixedPointTypes.fm_t)

        val scanBOutput = runSSM(
          name = "SSM_B",
          u = convBOutput,
          delta = dtProjBOutput,
          z_silu = flipZOutput,
          B = bBOutput,
          C = cBOutput,
          numPatches = NUM_PATCHES,
          dInner = D_INNER,
          dState = D_STATE,
          scanDim = SCAN_DIM,
          addrDst = addrScanBOut,
          addrU = addr_ssm_u,
          addrDelta = addr_ssm_delta,
          addrZSilu = addr_ssm_z_silu,
          addrB = addr_ssm_B,
          addrC = addr_ssm_C,
          addrA = addrScanBWeights,
          addrD = addrScanBWeights + 0x100000L,
          A_data = aBData,
          D_data = dBData,
          refFile = Files.layerRef(layerIdx, "mixer.out_z_b.float32.bin")
        )

        val addrScanBFlipped = 0x00D00000L
        val scanBFlippedOutput = runPatchOps(
          name = "Flip scan_b (PATCH_OPS)",
          mode = PATCH_OP_FLIP,
          inputData = scanBOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addrScanBOut,
          addrDst = addrScanBFlipped
        )

        val addrSumOpsInB = addr_scan_out + 0x1000000L
        val sumDivideOutput = runNormSum(
          name = "Sum Divide Output (NORM_SUM)",
          mode = NORM_SUM_DIV2_ONLY,
          inputA = outZOutput,
          inputB = scanBFlippedOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrcA = addr_scan_out,
          addrSrcB = addrSumOpsInB,
          addrDst = addrSumDivideOut,
          addrWeights = 0,
          weights = Array.empty,
          refFile = Files.layerRef(layerIdx, "mixer.out_proj.input.float32.bin")
        )

        banner("LAYER: OUT_PROJ")

        val outProjSmoothScalesFloat = read_float_file(Files.layerBin(layerIdx, "mixer.out_proj.smooth_scales.float32.bin"), D_INNER)
        val outProjSmoothScalesFixed = floatArrayToFixed(outProjSmoothScalesFloat, FixedPointTypes.wt_linear_ss_t)

        val outProjSmoothOutput = runSmooth(
          name = "SMOOTH(out_proj)",
          inputData = sumDivideOutput,
          numPatches = NUM_PATCHES,
          dim = D_INNER,
          addrSrc = addr_outproj_smooth_in,
          addrDst = addr_outproj_smooth_out,
          addrWeights = addr_smooth_weights,
          scales = outProjSmoothScalesFixed
        )

        val outProjTotalWeightElems = D_MODEL * D_INNER
        val outProjWeightsInt4 = read_int4_file(Files.layerBin(layerIdx, "mixer.out_proj.weight.int4.bin"), outProjTotalWeightElems)
        val outProjScalesPerRow = (D_INNER + 31) / 32
        val outProjScalesFixed = readFloatToFixed(
          Files.layerBin(layerIdx, "mixer.out_proj.weight_scale.float32.bin"),
          D_MODEL * outProjScalesPerRow,
          FixedPointTypes.wt_linear_ws_t
        )

        val outProjOutput = runLinear(
          name = "LINEAR(out_proj)",
          inputData = outProjSmoothOutput,
          numPatches = NUM_PATCHES,
          inDim = D_INNER,
          outDim = D_MODEL,
          flags = 0,
          addrSrc = addr_outproj_smooth_out,
          addrDst = addr_outproj_out,
          addrWeightsBase = addr_linear_outproj_weights,
          weightsInt4 = outProjWeightsInt4,
          biasOpt = None,
          scales = outProjScalesFixed,
          refFile = Files.layerRef(layerIdx, "mixer.out_proj.output.float32.bin")
        )

        banner("LAYER: RESIDUAL ADD")

        val addrSumOpsResidA = 0x20000000L
        val addrSumOpsResidB = addrSumOpsResidA + 0x1000000L
        val mixerOutput = runNormSum(
          name = "Last Residual Output (NORM_SUM(residual))",
          mode = NORM_SUM_ADD_ONLY,
          inputA = outProjOutput,
          inputB = normInput,
          numPatches = NUM_PATCHES,
          dim = D_MODEL,
          addrSrcA = addrSumOpsResidA,
          addrSrcB = addrSumOpsResidB,
          addrDst = addr_mixer_out,
          addrWeights = 0,
          weights = Array.empty,
          refFile = Files.refFloat32("last_residual.float32.bin")
        )

        lastLayerOutput = mixerOutput
        layerInput = mixerOutput
      }

      lastLayerOutput = expectNonEmpty(lastLayerOutput, "lastLayerOutput")

      banner("POST: LOAD_CLS_TOKEN (PATCH_OPS)")

      val PADDED_NUM_CLASSES = 1008
      val addrHeadOut = 0x17000000L
      val addrLinearHeadWeights = 0x20000000L
      val addrClsIn = 0x00E00000L
      val addrClsOut = 0x00F00000L

      val clsToken = runPatchOps(
        name = "PATCH_OPS(load_cls)",
        mode = 1,
        inputData = lastLayerOutput,
        numPatches = NUM_PATCHES,
        dim = D_MODEL,
        addrSrc = addrClsIn,
        addrDst = addrClsOut,
        refFile = Files.refFloat32("norm_f.input.float32.bin"),
        refNumPatches = 1
      )

      banner("POST: FINAL NORM")

      val finalNormWeightsFixed = readFloatToFixed(Files.binFloat32("norm_f.weight.float32.bin"), D_MODEL, FixedPointTypes.wt_norm_t)
      val finalNormOutput = runNormSum(
        name = "FINAL_NORM",
        mode = NORM_SUM_NORM_ONLY,
        inputA = clsToken,
        inputB = Array.empty,
        numPatches = 1,
        dim = D_MODEL,
        addrSrcA = addr_input,
        addrSrcB = 0,
        addrDst = addr_norm_out,
        addrWeights = addr_norm_weights,
        weights = finalNormWeightsFixed,
        refFile = Files.refFloat32("norm_f.output.float32.bin")
      )

      banner("POST: HEAD (LINEAR)")

      val headTotalWeightElems = PADDED_NUM_CLASSES * D_MODEL
      val headWeightsInt4 = read_int4_file(Files.binFloat32("head.weight.int4.bin"), headTotalWeightElems)
      val headScalesFixed = readFloatToFixed(
        Files.binFloat32("head.weight_scale.float32.bin"),
        PADDED_NUM_CLASSES,
        FixedPointTypes.wt_linear_ws_t
      )
      val headBiasFixed = readFloatToFixed(Files.binFloat32("head.bias.float32.bin"), PADDED_NUM_CLASSES, FixedPointTypes.wt_linear_bias_t)

      val headOutput = runLinear(
        name = "HEAD",
        inputData = finalNormOutput,
        numPatches = 1,
        inDim = D_MODEL,
        outDim = PADDED_NUM_CLASSES,
        flags = 1,
        addrSrc = addr_norm_out,
        addrDst = addrHeadOut,
        addrWeightsBase = addrLinearHeadWeights,
        weightsInt4 = headWeightsInt4,
        biasOpt = Some(headBiasFixed),
        scales = headScalesFixed,
        refFile = Files.refFloat32("head.output.float32.bin")
      )

      val endTime = simTime()
      val totalTime = (endTime - startTime) / SIM_CLOCK_PERIOD
      val totalTimeUs = totalTime / CLOCK_FREQ_MHZ
      val totalTimeMs = totalTimeUs / 1000.0

      val latencySummary = LatencyTracker.generateReport(
        "latency_report.log",
        CLOCK_FREQ_MHZ,
        SIM_CLOCK_PERIOD,
        simulatedLayers = NUM_LAYERS,
        modelLayers = MODEL_NUM_LAYERS,
        preLayerNames = preLayerNames,
        postLayerNames = postLayerNames
      )

      val estTotalTimeMs = latencySummary.totalE2ETimeMsRoundedSum
      val throughputEst = if (estTotalTimeMs > 0.0) {
        NUM_PATCHES.toDouble / estTotalTimeMs
      } else {
        0.0
      }

      println("\n" + "=" * 80)
      println("=== TEST SUMMARY ===")
      println("=" * 80)
      println(f"\n[PERFORMANCE] Full Pipeline:")
      println(f"  Model Layers: $MODEL_NUM_LAYERS")
      println(f"  Pre-layer Cycles (once): $preLayerCycles")
      println(f"  In-layer Cycles (sim): $inLayerCycles")
      println(f"  Post-layer Cycles (once): $postLayerCycles")
      println(f"  Total Cycles (sim run): $totalCycles")
      println(f"  Total Time (e2e): $estTotalTimeMs%.3f ms")
      println(f"  Throughput (e2e): $throughputEst%.2f patches/ms")

      dut.clockDomain.waitSampling(100)
      simSuccess()
    }
}
