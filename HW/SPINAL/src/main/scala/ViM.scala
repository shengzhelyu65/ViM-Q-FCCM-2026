import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.Axi4SpecRenamer
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import utils._

import scala.language.postfixOps

class ViM extends Component {
  // Instantiate modules once
  val embed = new EMBED()
  val norm_sum = new NORM_SUM()
  val linear = new LINEAR_BLOCK()
  val conv   = new CONV()
  val smooth = new SMOOTH()
  val ssm    = new SSM()
  val patch_ops = new PATCH_OPS()

  val io = new Bundle {
    // DaisyChain for sequential control
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    
    // AXI4 memory interfaces - all modules share same DRAM space
    // EMBED interfaces
    val m_axi_embed_out_r: Axi4 = master(Axi4(embed.io.m_axi_out_r.config))
    val m_axi_embed_in_r: Axi4 = master(Axi4(embed.io.m_axi_in_r.config))
    val m_axi_embed_weights: Axi4 = master(Axi4(embed.io.m_axi_weights.config))
    
    // NORM_SUM interfaces
    val m_axi_norm_sum_in_a: Axi4 = master(Axi4(norm_sum.io.m_axi_in_a.config))
    val m_axi_norm_sum_in_b: Axi4 = master(Axi4(norm_sum.io.m_axi_in_b.config))
    val m_axi_norm_sum_out_r: Axi4 = master(Axi4(norm_sum.io.m_axi_out_r.config))
    val m_axi_norm_sum_weights: Axi4 = master(Axi4(norm_sum.io.m_axi_weights.config))
    
    // LINEAR interfaces
    val m_axi_linear_in: Axi4 = master(Axi4(linear.io.m_axi_in_r.config))
    val m_axi_linear_out: Axi4 = master(Axi4(linear.io.m_axi_out_r.config))
    val m_axi_linear_weights: Axi4 = master(Axi4(linear.io.m_axi_weights.config))
    
    // CONV interfaces
    val m_axi_conv_in: Axi4 = master(Axi4(conv.io.m_axi_in_r.config))
    val m_axi_conv_out: Axi4 = master(Axi4(conv.io.m_axi_out_r.config))
    val m_axi_conv_weights: Axi4 = master(Axi4(conv.io.m_axi_weights.config))
    
    // SMOOTH interfaces
    val m_axi_smooth_in: Axi4 = master(Axi4(smooth.io.m_axi_in_r.config))
    val m_axi_smooth_out: Axi4 = master(Axi4(smooth.io.m_axi_out_r.config))
    val m_axi_smooth_weights: Axi4 = master(Axi4(smooth.io.m_axi_weights.config))
    
    val m_axi_ssm_in_u: Axi4 = master(Axi4(ssm.io.m_axi_in_u.config))
    val m_axi_ssm_in_delta: Axi4 = master(Axi4(ssm.io.m_axi_in_delta.config))
    val m_axi_ssm_in_z_silu: Axi4 = master(Axi4(ssm.io.m_axi_in_z_silu.config))
    val m_axi_ssm_in_B: Axi4 = master(Axi4(ssm.io.m_axi_in_B.config))
    val m_axi_ssm_in_C: Axi4 = master(Axi4(ssm.io.m_axi_in_C.config))
    val m_axi_ssm_weights_A: Axi4 = master(Axi4(ssm.io.m_axi_weights_A.config))
    val m_axi_ssm_weights_D: Axi4 = master(Axi4(ssm.io.m_axi_weights_D.config))
    val m_axi_ssm_out_r: Axi4 = master(Axi4(ssm.io.m_axi_out_r.config))
    
    // PATCH_OPS interfaces
    val m_axi_patch_ops_in: Axi4 = master(Axi4(patch_ops.io.m_axi_in_r.config))
    val m_axi_patch_ops_out: Axi4 = master(Axi4(patch_ops.io.m_axi_out_r.config))
    
    // AXI-Lite control interfaces - software configures each module
    // EMBED control
    val s_axi_embed_control: AxiLite4 = slave(AxiLite4(embed.io.s_axi_control.config))
    val s_axi_embed_control_r: AxiLite4 = slave(AxiLite4(embed.io.s_axi_control_r.config))
    
    // NORM_SUM control
    val s_axi_norm_sum_control: AxiLite4 = slave(AxiLite4(norm_sum.io.s_axi_control.config))
    val s_axi_norm_sum_control_r: AxiLite4 = slave(AxiLite4(norm_sum.io.s_axi_control_r.config))
    
    // LINEAR control
    val s_axi_linear_control: AxiLite4 = slave(AxiLite4(linear.io.s_axi_control.config))
    val s_axi_linear_control_r: AxiLite4 = slave(AxiLite4(linear.io.s_axi_control_r.config))
    
    // CONV control
    val s_axi_conv_control: AxiLite4 = slave(AxiLite4(conv.io.s_axi_control.config))
    val s_axi_conv_control_r: AxiLite4 = slave(AxiLite4(conv.io.s_axi_control_r.config))
    
    // SMOOTH control
    val s_axi_smooth_control: AxiLite4 = slave(AxiLite4(smooth.io.s_axi_control.config))
    val s_axi_smooth_control_r: AxiLite4 = slave(AxiLite4(smooth.io.s_axi_control_r.config))
    
    val s_axi_ssm_control: AxiLite4 = slave(AxiLite4(ssm.io.s_axi_control.config))
    val s_axi_ssm_control_r: AxiLite4 = slave(AxiLite4(ssm.io.s_axi_control_r.config))
    
    // PATCH_OPS control
    val s_axi_patch_ops_control: AxiLite4 = slave(AxiLite4(patch_ops.io.s_axi_control.config))
    val s_axi_patch_ops_control_r: AxiLite4 = slave(AxiLite4(patch_ops.io.s_axi_control_r.config))
  }
  noIoPrefix()

  // Connect DaisyChain for sequential control
  io.signals.I        <> embed.io.signals.I
  embed.io.signals.O  <> norm_sum.io.signals.I
  norm_sum.io.signals.O <> linear.io.signals.I
  linear.io.signals.O <> conv.io.signals.I
  conv.io.signals.O   <> smooth.io.signals.I
  smooth.io.signals.O <> ssm.io.signals.I
  ssm.io.signals.O    <> io.signals.O
  
  // Connect AXI4 memory interfaces - expose separately
  // In Vivado: Connect all to shared DRAM via AXI Interconnect
  embed.io.m_axi_out_r     <> io.m_axi_embed_out_r
  embed.io.m_axi_in_r      <> io.m_axi_embed_in_r
  embed.io.m_axi_weights   <> io.m_axi_embed_weights
  
  norm_sum.io.m_axi_in_a     <> io.m_axi_norm_sum_in_a
  norm_sum.io.m_axi_in_b     <> io.m_axi_norm_sum_in_b
  norm_sum.io.m_axi_out_r    <> io.m_axi_norm_sum_out_r
  norm_sum.io.m_axi_weights  <> io.m_axi_norm_sum_weights
  
  linear.io.m_axi_in_r   <> io.m_axi_linear_in
  linear.io.m_axi_out_r  <> io.m_axi_linear_out
  linear.io.m_axi_weights <> io.m_axi_linear_weights
  
  conv.io.m_axi_in_r     <> io.m_axi_conv_in
  conv.io.m_axi_out_r    <> io.m_axi_conv_out
  conv.io.m_axi_weights  <> io.m_axi_conv_weights
  
  smooth.io.m_axi_in_r   <> io.m_axi_smooth_in
  smooth.io.m_axi_out_r  <> io.m_axi_smooth_out
  smooth.io.m_axi_weights <> io.m_axi_smooth_weights
  
  ssm.io.m_axi_in_u      <> io.m_axi_ssm_in_u
  ssm.io.m_axi_in_delta  <> io.m_axi_ssm_in_delta
  ssm.io.m_axi_in_z_silu <> io.m_axi_ssm_in_z_silu
  ssm.io.m_axi_in_B      <> io.m_axi_ssm_in_B
  ssm.io.m_axi_in_C      <> io.m_axi_ssm_in_C
  ssm.io.m_axi_weights_A <> io.m_axi_ssm_weights_A
  ssm.io.m_axi_weights_D <> io.m_axi_ssm_weights_D
  ssm.io.m_axi_out_r     <> io.m_axi_ssm_out_r
  
  patch_ops.io.m_axi_in_r  <> io.m_axi_patch_ops_in
  patch_ops.io.m_axi_out_r <> io.m_axi_patch_ops_out

  // Connect AXI-Lite control interfaces - expose separately
  // In Vivado: Connect all to CPU via AXI SmartConnect with address decoding
  embed.io.s_axi_control     <> io.s_axi_embed_control
  embed.io.s_axi_control_r    <> io.s_axi_embed_control_r
  
  norm_sum.io.s_axi_control     <> io.s_axi_norm_sum_control
  norm_sum.io.s_axi_control_r    <> io.s_axi_norm_sum_control_r
  
  linear.io.s_axi_control   <> io.s_axi_linear_control
  linear.io.s_axi_control_r <> io.s_axi_linear_control_r
  
  conv.io.s_axi_control     <> io.s_axi_conv_control
  conv.io.s_axi_control_r   <> io.s_axi_conv_control_r
  
  smooth.io.s_axi_control   <> io.s_axi_smooth_control
  smooth.io.s_axi_control_r <> io.s_axi_smooth_control_r
  
  ssm.io.s_axi_control      <> io.s_axi_ssm_control
  ssm.io.s_axi_control_r    <> io.s_axi_ssm_control_r
  
  patch_ops.io.s_axi_control   <> io.s_axi_patch_ops_control
  patch_ops.io.s_axi_control_r <> io.s_axi_patch_ops_control_r
  
  // Apply AXI naming conventions for Vivado compatibility
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
