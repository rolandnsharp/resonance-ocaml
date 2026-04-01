(** One layer: wave-field attention + FFN, with full backward pass.

    Forward stores intermediates. Backward computes gradients
    and updates weights. Each layer is self-contained —
    can run on a separate RISC-V node. *)

(* --- Utilities --- *)

let rms_norm x =
  let n = Float.of_int (Array.length x) in
  let rms = sqrt (Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x /. n +. 1e-8) in
  Array.map (fun xi -> xi /. rms) x

let mat_vec w ~rows ~cols x =
  Array.init rows (fun i ->
    let acc = ref 0.0 in
    let base = i * cols in
    for j = 0 to cols - 1 do
      acc := !acc +. w.(base + j) *. x.(j)
    done; !acc)

(** Transposed mat-vec: W^T × x *)
let mat_vec_t w ~rows ~cols x =
  Array.init cols (fun j ->
    let acc = ref 0.0 in
    for i = 0 to rows - 1 do
      acc := !acc +. w.(i * cols + j) *. x.(i)
    done; !acc)

(** Accumulate outer product into gradient: grad += scale * a ⊗ b *)
let outer_accum grad a b ~cols ~scale =
  Array.iteri (fun i ai ->
    if Float.abs ai > 1e-8 then
      let lr_ai = scale *. ai in
      let base = i * cols in
      Array.iteri (fun j bj ->
        grad.(base + j) <- grad.(base + j) +. lr_ai *. bj
      ) b
  ) a

let sigmoid x = 1.0 /. (1.0 +. exp (-. x))

let gelu x =
  0.5 *. x *. (1.0 +. tanh (0.7978846 *. (x +. 0.044715 *. x *. x *. x)))

let gelu_deriv x =
  let t = tanh (0.7978846 *. (x +. 0.044715 *. x *. x *. x)) in
  let dtanh = 1.0 -. t *. t in
  0.5 *. (1.0 +. t) +. 0.5 *. x *. dtanh *. 0.7978846 *. (1.0 +. 3.0 *. 0.044715 *. x *. x)

(* --- Layer --- *)

type t = {
  dim : int;
  hidden_dim : int;
  w_gate : float array;     (** dim × dim: content gate *)
  w_up : float array;       (** hidden_dim × dim *)
  w_down : float array;     (** dim × hidden_dim *)
  n_osc : int;
  oscillators : Oscillator.t array;
  mutable kernels : Bank.kernels;
}

let create ~dim ~n_osc ~seq_len =
  let hidden_dim = dim * 2 in
  let scale_d = 1.0 /. sqrt (Float.of_int dim) in
  let scale_h = 1.0 /. sqrt (Float.of_int hidden_dim) in
  let oscillators = Oscillator.spread n_osc in
  {
    dim; hidden_dim;
    w_gate = Array.init (dim * dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale_d *. 0.01);
    w_up = Array.init (hidden_dim * dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale_d);
    w_down = Array.init (dim * hidden_dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale_h);
    n_osc; oscillators;
    kernels = Bank.precompute_kernels oscillators seq_len;
  }

(* --- Forward: stores intermediates for backward --- *)

type position_cache = {
  input : float array;
  normed1 : float array;
  gate_pre : float array;
  gate_val : float array;
  gated : float array;
  bank_out : float array;
  after_wf : float array;
  normed2 : float array;
  ffn_hidden : float array;
  ffn_activated : float array;
}

(** Forward one position through FFN only (wave-field is per-sequence) *)
let ffn_forward layer x =
  let h = mat_vec layer.w_up ~rows:layer.hidden_dim ~cols:layer.dim x in
  let a = Array.map gelu h in
  let out = mat_vec layer.w_down ~rows:layer.dim ~cols:layer.hidden_dim a in
  (h, a, out)

(** Forward the full sequence through this layer *)
let forward layer (states : float array array) =
  let seq_len = Array.length states in
  let dim = layer.dim in
  let n_osc = layer.n_osc in

  (* Wave field: gate the input, FFT convolve, apply as residual *)
  let normed1 = Array.map rms_norm states in

  let gate_pres = Array.map (fun s ->
    mat_vec layer.w_gate ~rows:dim ~cols:dim s
  ) normed1 in
  let gate_vals = Array.map (fun pre ->
    Array.map sigmoid pre
  ) gate_pres in
  let gated = Array.map2 (fun g s ->
    Array.map2 ( *. ) g s
  ) gate_vals normed1 in

  (* FFT convolve the gated input *)
  let drives = Array.map (fun g ->
    Array.init n_osc (fun i -> g.(i))
  ) gated in
  let bank_outs = Bank.encode layer.oscillators layer.kernels drives in

  (* Residual after wave field *)
  let after_wf = Array.map2 (fun s b ->
    Array.init dim (fun i ->
      s.(i) +. b.(i mod (2 * n_osc))
    )
  ) states bank_outs in

  (* FFN with residual *)
  let normed2 = Array.map rms_norm after_wf in
  let caches = Array.init seq_len (fun t ->
    let h, a, ffn_out = ffn_forward layer normed2.(t) in
    let output = Array.map2 ( +. ) after_wf.(t) ffn_out in
    (output, {
      input = states.(t);
      normed1 = normed1.(t);
      gate_pre = gate_pres.(t);
      gate_val = gate_vals.(t);
      gated = gated.(t);
      bank_out = bank_outs.(t);
      after_wf = after_wf.(t);
      normed2 = normed2.(t);
      ffn_hidden = h;
      ffn_activated = a;
    })
  ) in
  let outputs = Array.map fst caches in
  let cache = Array.map snd caches in
  (outputs, cache)

(* --- Backward --- *)

(** Backward through FFN: returns d_input, updates w_up and w_down *)
let ffn_backward layer cache d_output ~lr =
  let dim = layer.dim in
  let hdim = layer.hidden_dim in
  (* d_activated = w_down^T × d_output *)
  let d_activated = mat_vec_t layer.w_down ~rows:dim ~cols:hdim d_output in
  (* Update w_down: d_output ⊗ activated *)
  outer_accum layer.w_down d_output cache.ffn_activated ~cols:hdim ~scale:(-.lr);
  (* d_hidden = d_activated × gelu'(hidden) *)
  let d_hidden = Array.map2 (fun da h -> da *. gelu_deriv h) d_activated cache.ffn_hidden in
  (* Update w_up: d_hidden ⊗ normed2 *)
  outer_accum layer.w_up d_hidden cache.normed2 ~cols:dim ~scale:(-.lr);
  (* d_normed2 = w_up^T × d_hidden *)
  mat_vec_t layer.w_up ~rows:hdim ~cols:dim d_hidden

(** Backward through gate: returns d_normed1, updates w_gate *)
let gate_backward layer cache d_gated ~lr =
  let dim = layer.dim in
  (* d_gate = d_gated × normed1 × sigmoid'(gate_pre) *)
  let d_gate = Array.init dim (fun i ->
    d_gated.(i) *. cache.normed1.(i) *. cache.gate_val.(i) *. (1.0 -. cache.gate_val.(i))
  ) in
  (* Update w_gate *)
  outer_accum layer.w_gate d_gate cache.normed1 ~cols:dim ~scale:(-.lr);
  (* d_normed1 from gate = d_gated × gate_val *)
  Array.map2 (fun dg gv -> dg *. gv) d_gated cache.gate_val

(** Full layer backward: returns d_input for previous layer *)
let backward layer cache d_output ~lr =
  (* Through FFN residual: gradient flows to both FFN and after_wf *)
  let d_ffn = d_output in
  let d_after_wf_from_ffn = d_output in

  (* Backprop through FFN *)
  let d_normed2 = ffn_backward layer cache d_ffn ~lr in
  (* RMSNorm backward (approximate: just pass through scaled) *)
  let d_after_wf = Array.map2 ( +. ) d_after_wf_from_ffn d_normed2 in

  (* Through wave-field residual *)
  let d_bank = d_after_wf in
  let d_input_from_wf = d_after_wf in

  (* Approximate: gradient through bank → gate.
     Full FFT backward would be another FFT, but for now pass gradient through *)
  let d_gated = Array.init layer.dim (fun i ->
    d_bank.(i mod (2 * layer.n_osc))
  ) in

  (* Backprop through gate *)
  let d_normed1 = gate_backward layer cache d_gated ~lr in

  (* Combine: d_input from residual + d_input from wave_field path *)
  Array.map2 ( +. ) d_input_from_wf d_normed1

(** Forward + backward for full sequence. Returns (outputs, d_inputs) *)
let train layer states d_outputs ~lr =
  let outputs, cache = forward layer states in
  let d_inputs = Array.map2 (fun c d_out ->
    backward layer c d_out ~lr
  ) cache d_outputs in
  (outputs, d_inputs)
