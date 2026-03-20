import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.Axi4SpecRenamer
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.core.sim._
import utils._

import scala.language.postfixOps

/**
 * ViM Accelerator - Top-Level Component
 * 
 * Wraps the ViM pipeline for system integration.
 * This is the public interface that remains stable while internal ViM implementation can evolve.
 */
class ViM_ACCELERATOR extends Component {
  val inst_vim = new ViM()

  val io = new Bundle {
    // DaisyChain for control flow
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    // AXI4 memory interfaces - all connect to shared DRAM
    val m_axi_embed_out_r: Axi4 = master(Axi4(inst_vim.io.m_axi_embed_out_r.config))
    val m_axi_embed_in_r: Axi4 = master(Axi4(inst_vim.io.m_axi_embed_in_r.config))
    val m_axi_embed_weights: Axi4 = master(Axi4(inst_vim.io.m_axi_embed_weights.config))
    val m_axi_norm_sum_in_a: Axi4 = master(Axi4(inst_vim.io.m_axi_norm_sum_in_a.config))
    val m_axi_norm_sum_in_b: Axi4 = master(Axi4(inst_vim.io.m_axi_norm_sum_in_b.config))
    val m_axi_norm_sum_out_r: Axi4 = master(Axi4(inst_vim.io.m_axi_norm_sum_out_r.config))
    val m_axi_norm_sum_weights: Axi4 = master(Axi4(inst_vim.io.m_axi_norm_sum_weights.config))
    val m_axi_linear_in: Axi4 = master(Axi4(inst_vim.io.m_axi_linear_in.config))
    val m_axi_linear_out: Axi4 = master(Axi4(inst_vim.io.m_axi_linear_out.config))
    val m_axi_linear_weights: Axi4 = master(Axi4(inst_vim.io.m_axi_linear_weights.config))
    val m_axi_conv_in: Axi4 = master(Axi4(inst_vim.io.m_axi_conv_in.config))
    val m_axi_conv_out: Axi4 = master(Axi4(inst_vim.io.m_axi_conv_out.config))
    val m_axi_conv_weights: Axi4 = master(Axi4(inst_vim.io.m_axi_conv_weights.config))
    val m_axi_smooth_in: Axi4 = master(Axi4(inst_vim.io.m_axi_smooth_in.config))
    val m_axi_smooth_out: Axi4 = master(Axi4(inst_vim.io.m_axi_smooth_out.config))
    val m_axi_smooth_weights: Axi4 = master(Axi4(inst_vim.io.m_axi_smooth_weights.config))
    val m_axi_ssm_in_u: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_in_u.config))
    val m_axi_ssm_in_delta: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_in_delta.config))
    val m_axi_ssm_in_z_silu: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_in_z_silu.config))
    val m_axi_ssm_in_B: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_in_B.config))
    val m_axi_ssm_in_C: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_in_C.config))
    val m_axi_ssm_weights_A: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_weights_A.config))
    val m_axi_ssm_weights_D: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_weights_D.config))
    val m_axi_ssm_out_r: Axi4 = master(Axi4(inst_vim.io.m_axi_ssm_out_r.config))
    val m_axi_patch_ops_in: Axi4 = master(Axi4(inst_vim.io.m_axi_patch_ops_in.config))
    val m_axi_patch_ops_out: Axi4 = master(Axi4(inst_vim.io.m_axi_patch_ops_out.config))
    
    // AXI-Lite control interfaces - all connect to CPU via AXI SmartConnect
    val s_axi_embed_control: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_embed_control.config))
    val s_axi_embed_control_r: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_embed_control_r.config))
    val s_axi_norm_sum_control: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_norm_sum_control.config))
    val s_axi_norm_sum_control_r: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_norm_sum_control_r.config))
    val s_axi_linear_control: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_linear_control.config))
    val s_axi_linear_control_r: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_linear_control_r.config))
    val s_axi_conv_control: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_conv_control.config))
    val s_axi_conv_control_r: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_conv_control_r.config))
    val s_axi_smooth_control: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_smooth_control.config))
    val s_axi_smooth_control_r: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_smooth_control_r.config))
    val s_axi_ssm_control: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_ssm_control.config))
    val s_axi_ssm_control_r: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_ssm_control_r.config))
    val s_axi_patch_ops_control: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_patch_ops_control.config))
    val s_axi_patch_ops_control_r: AxiLite4 = slave(AxiLite4(inst_vim.io.s_axi_patch_ops_control_r.config))
  }
  noIoPrefix()

  // Pass-through all connections
  io.signals <> inst_vim.io.signals
  
  io.m_axi_embed_out_r      <> inst_vim.io.m_axi_embed_out_r
  io.m_axi_embed_in_r       <> inst_vim.io.m_axi_embed_in_r
  io.m_axi_embed_weights    <> inst_vim.io.m_axi_embed_weights
  io.m_axi_norm_sum_in_a     <> inst_vim.io.m_axi_norm_sum_in_a
  io.m_axi_norm_sum_in_b     <> inst_vim.io.m_axi_norm_sum_in_b
  io.m_axi_norm_sum_out_r    <> inst_vim.io.m_axi_norm_sum_out_r
  io.m_axi_norm_sum_weights  <> inst_vim.io.m_axi_norm_sum_weights
  io.m_axi_linear_in   <> inst_vim.io.m_axi_linear_in
  io.m_axi_linear_out  <> inst_vim.io.m_axi_linear_out
  io.m_axi_linear_weights <> inst_vim.io.m_axi_linear_weights
  io.m_axi_conv_in     <> inst_vim.io.m_axi_conv_in
  io.m_axi_conv_out    <> inst_vim.io.m_axi_conv_out
  io.m_axi_conv_weights <> inst_vim.io.m_axi_conv_weights
  io.m_axi_smooth_in   <> inst_vim.io.m_axi_smooth_in
  io.m_axi_smooth_out  <> inst_vim.io.m_axi_smooth_out
  io.m_axi_smooth_weights <> inst_vim.io.m_axi_smooth_weights
  io.m_axi_ssm_in_u       <> inst_vim.io.m_axi_ssm_in_u
  io.m_axi_ssm_in_delta   <> inst_vim.io.m_axi_ssm_in_delta
  io.m_axi_ssm_in_z_silu  <> inst_vim.io.m_axi_ssm_in_z_silu
  io.m_axi_ssm_in_B       <> inst_vim.io.m_axi_ssm_in_B
  io.m_axi_ssm_in_C       <> inst_vim.io.m_axi_ssm_in_C
  io.m_axi_ssm_weights_A  <> inst_vim.io.m_axi_ssm_weights_A
  io.m_axi_ssm_weights_D  <> inst_vim.io.m_axi_ssm_weights_D
  io.m_axi_ssm_out_r      <> inst_vim.io.m_axi_ssm_out_r
  io.m_axi_patch_ops_in <> inst_vim.io.m_axi_patch_ops_in
  io.m_axi_patch_ops_out <> inst_vim.io.m_axi_patch_ops_out
  
  io.s_axi_embed_control      <> inst_vim.io.s_axi_embed_control
  io.s_axi_embed_control_r    <> inst_vim.io.s_axi_embed_control_r
  io.s_axi_norm_sum_control     <> inst_vim.io.s_axi_norm_sum_control
  io.s_axi_norm_sum_control_r   <> inst_vim.io.s_axi_norm_sum_control_r
  io.s_axi_linear_control   <> inst_vim.io.s_axi_linear_control
  io.s_axi_linear_control_r <> inst_vim.io.s_axi_linear_control_r
  io.s_axi_conv_control     <> inst_vim.io.s_axi_conv_control
  io.s_axi_conv_control_r   <> inst_vim.io.s_axi_conv_control_r
  io.s_axi_smooth_control   <> inst_vim.io.s_axi_smooth_control
  io.s_axi_smooth_control_r <> inst_vim.io.s_axi_smooth_control_r
  io.s_axi_ssm_control     <> inst_vim.io.s_axi_ssm_control
  io.s_axi_ssm_control_r   <> inst_vim.io.s_axi_ssm_control_r
  io.s_axi_patch_ops_control <> inst_vim.io.s_axi_patch_ops_control
  io.s_axi_patch_ops_control_r <> inst_vim.io.s_axi_patch_ops_control_r
  
  // Apply AXI naming conventions
  Axi4SpecRenamer(io.m_axi_embed_out_r)
  Axi4SpecRenamer(io.m_axi_embed_in_r)
  Axi4SpecRenamer(io.m_axi_embed_weights)
  Axi4SpecRenamer(io.m_axi_norm_sum_in_a)
  Axi4SpecRenamer(io.m_axi_norm_sum_in_b)
  Axi4SpecRenamer(io.m_axi_norm_sum_out_r)
  Axi4SpecRenamer(io.m_axi_norm_sum_weights)
  Axi4SpecRenamer(io.m_axi_linear_in)
  Axi4SpecRenamer(io.m_axi_linear_out)
  Axi4SpecRenamer(io.m_axi_linear_weights)
  Axi4SpecRenamer(io.m_axi_conv_in)
  Axi4SpecRenamer(io.m_axi_conv_out)
  Axi4SpecRenamer(io.m_axi_conv_weights)
  Axi4SpecRenamer(io.m_axi_smooth_in)
  Axi4SpecRenamer(io.m_axi_smooth_out)
  Axi4SpecRenamer(io.m_axi_smooth_weights)
  Axi4SpecRenamer(io.m_axi_ssm_in_u)
  Axi4SpecRenamer(io.m_axi_ssm_in_delta)
  Axi4SpecRenamer(io.m_axi_ssm_in_z_silu)
  Axi4SpecRenamer(io.m_axi_ssm_in_B)
  Axi4SpecRenamer(io.m_axi_ssm_in_C)
  Axi4SpecRenamer(io.m_axi_ssm_weights_A)
  Axi4SpecRenamer(io.m_axi_ssm_weights_D)
  Axi4SpecRenamer(io.m_axi_ssm_out_r)
  Axi4SpecRenamer(io.m_axi_patch_ops_in)
  Axi4SpecRenamer(io.m_axi_patch_ops_out)
  
  // Apply AXI Lite naming conventions
  AxiLite4SpecRenamer(io.s_axi_embed_control)
  AxiLite4SpecRenamer(io.s_axi_embed_control_r)
  AxiLite4SpecRenamer(io.s_axi_norm_sum_control)
  AxiLite4SpecRenamer(io.s_axi_norm_sum_control_r)
  AxiLite4SpecRenamer(io.s_axi_linear_control)
  AxiLite4SpecRenamer(io.s_axi_linear_control_r)
  AxiLite4SpecRenamer(io.s_axi_conv_control)
  AxiLite4SpecRenamer(io.s_axi_conv_control_r)
  AxiLite4SpecRenamer(io.s_axi_smooth_control)
  AxiLite4SpecRenamer(io.s_axi_smooth_control_r)
  AxiLite4SpecRenamer(io.s_axi_ssm_control)
  AxiLite4SpecRenamer(io.s_axi_ssm_control_r)
  AxiLite4SpecRenamer(io.s_axi_patch_ops_control)
  AxiLite4SpecRenamer(io.s_axi_patch_ops_control_r)
}

/**
 * Generate Verilog for ViM_ACCELERATOR
 */
object generate_vim_accelerator extends App {
  val spinalConfig: SpinalConfig = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = SYNC, resetActiveLevel = LOW
    )
  )
  SpinalVerilog(spinalConfig)(new ViM_ACCELERATOR).mergeRTLSource()
}

