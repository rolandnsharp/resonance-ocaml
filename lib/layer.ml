(** One layer: wave-field attention + wave-native nonlinear mixing.

    No matrix multiplies. Every operation is wave-native:
    energy norm, interference gate, causal EMA, FFT bank,
    phase rotation, Duffing nonlinearity, spectral fold.

    Each layer is self-contained — one RISC-V node. *)

(* --- Utilities --- *)

let rms_norm x =
  let n = Float.of_int (Array.length x) in
  let rms = sqrt (Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x /. n +. 1e-8) in
  Array.map (fun xi -> xi /. rms) x

(** Energy normalization: normalize by total oscillator energy.
    Respects (position, velocity) conjugate pairs. *)
let energy_norm x n_osc =
  let e = ref 0.0 in
  for k = 0 to n_osc - 1 do
    e := !e +. x.(k) *. x.(k) +. x.(k + n_osc) *. x.(k + n_osc)
  done;
  let inv = 1.0 /. sqrt (!e /. Float.of_int n_osc +. 1e-8) in
  Array.map (fun xi -> xi *. inv) x

(** Clip vector to max norm *)
let clip_norm max_norm v =
  let norm = sqrt (Array.fold_left (fun acc x -> acc +. x *. x) 0.0 v) in
  if norm > max_norm then Array.map (fun x -> x *. max_norm /. norm) v
  else v

(* --- Layer --- *)

type t = {
  dim : int;
  fold_offset : int;           (** butterfly coupling: pairs k with (k+offset) mod n *)
  (* Wave-field: interference gate + causal feedback *)
  ref_phases : float array;    (** n_osc: gate reference phases *)
  causal_decay : float array;  (** n_osc: per-oscillator EMA β *)
  (* Wave mix: phase rotation + Duffing + spectral fold *)
  epsilon : float array;       (** n_osc: cubic nonlinearity strength *)
  coupling : float array;      (** n_osc: cross-spectral coupling *)
  rotation : float array;      (** n_osc: phase rotation angles *)
  scale : float array;         (** n_osc: output scaling *)
  (* Gradient accumulators *)
  d_ref_phases : float array;
  d_causal_decay : float array;
  d_epsilon : float array;
  d_coupling : float array;
  d_rotation : float array;
  d_scale : float array;
  n_osc : int;
  oscillators : Oscillator.t array;
  mutable kernels : Bank.kernels;
}

let create ~n_osc ~seq_len ~layer_idx =
  let n_levels = let rec go n = if n <= 1 then 0 else 1 + go (n / 2) in
    max 1 (go n_osc) in
  let oscillators = Oscillator.spread n_osc in
  {
    dim = 2 * n_osc;
    fold_offset = max 1 (n_osc lsr (1 + (layer_idx mod n_levels)));
    ref_phases = Array.make n_osc 0.0;
    causal_decay = Array.make n_osc 0.1;
    epsilon = Array.make n_osc 0.01;
    coupling = Array.make n_osc 0.1;
    rotation = Array.make n_osc 0.0;
    scale = Array.make n_osc 1.0;
    d_ref_phases = Array.make n_osc 0.0;
    d_causal_decay = Array.make n_osc 0.0;
    d_epsilon = Array.make n_osc 0.0;
    d_coupling = Array.make n_osc 0.0;
    d_rotation = Array.make n_osc 0.0;
    d_scale = Array.make n_osc 0.0;
    n_osc; oscillators;
    kernels = Bank.precompute_kernels oscillators seq_len;
  }

(* --- Forward --- *)

type position_cache = {
  input : float array;
  normed1 : float array;
  phases : float array;       (** atan2(v_k, x_k) per oscillator *)
  alignments : float array;   (** 0.5 + 0.5 * cos(phase - ref) *)
  causal_drive : float array;
  gated : float array;
  bank_out : float array;
  after_wf : float array;
  normed2 : float array;
  rotated : float array;      (** post-rotation, pre-Duffing *)
}

(** Wave-native nonlinear mixing:
    phase rotation → Duffing cubic → spectral fold → scale.
    Zero matrix multiplies. *)
let wave_mix_forward layer x =
  let n = layer.n_osc in
  let dim = layer.dim in
  (* Phase rotation: rotate each oscillator's (pos, vel) pair *)
  let rotated = Array.init dim (fun i ->
    let k = i mod n in
    let c = cos layer.rotation.(k) and s = sin layer.rotation.(k) in
    if i < n then x.(k) *. c -. x.(k + n) *. s
    else x.(i - n) *. s +. x.(i) *. c
  ) in
  (* Duffing nonlinearity + butterfly spectral fold *)
  let fold = layer.fold_offset in
  let out = Array.init dim (fun i ->
    let k = i mod n in
    let ri = rotated.(i) in
    let cubic = ri +. layer.epsilon.(k) *. ri *. ri *. ri in
    let partner = (k + fold) mod n in
    let pi = if i < n then partner else partner + n in
    layer.scale.(k) *. (cubic +. layer.coupling.(k) *. rotated.(pi))
  ) in
  (rotated, out)

(** Forward the full sequence through this layer *)
let forward layer (states : float array array) =
  let seq_len = Array.length states in
  let dim = layer.dim in
  let n_osc = layer.n_osc in

  (* Wave field: energy normalize, interference gate, causal EMA, FFT bank *)
  let normed1 = Array.map (fun s -> energy_norm s n_osc) states in

  let phases = Array.map (fun s ->
    Array.init n_osc (fun k -> atan2 s.(k + n_osc) s.(k))
  ) normed1 in
  let alignments = Array.map (fun ph ->
    Array.init n_osc (fun k ->
      0.5 +. 0.5 *. cos (ph.(k) -. layer.ref_phases.(k)))
  ) phases in
  let gated = Array.map2 (fun a s ->
    Array.init dim (fun i -> s.(i) *. a.(i mod n_osc))
  ) alignments normed1 in

  let drives = Array.map (fun g ->
    Array.init n_osc (fun i -> g.(i))
  ) gated in
  let causal_drives = Array.init seq_len (fun _ -> Array.make n_osc 0.0) in
  for k = 0 to n_osc - 1 do
    let beta = Float.max 0.0 (Float.min 0.95 layer.causal_decay.(k)) in
    causal_drives.(0).(k) <- drives.(0).(k);
    for t = 1 to seq_len - 1 do
      causal_drives.(t).(k) <-
        (1.0 -. beta) *. drives.(t).(k) +. beta *. causal_drives.(t-1).(k)
    done
  done;
  let bank_outs = Bank.encode layer.oscillators layer.kernels causal_drives in

  let after_wf = Array.map2 (fun s b ->
    Array.init dim (fun i -> s.(i) +. b.(i mod (2 * n_osc)))
  ) states bank_outs in

  (* Wave mix with residual *)
  let normed2 = Array.map (fun s -> energy_norm s n_osc) after_wf in
  let caches = Array.init seq_len (fun t ->
    let rotated, wave_out = wave_mix_forward layer normed2.(t) in
    let output = Array.map2 ( +. ) after_wf.(t) wave_out in
    (output, {
      input = states.(t);
      normed1 = normed1.(t);
      phases = phases.(t);
      alignments = alignments.(t);
      causal_drive = causal_drives.(t);
      gated = gated.(t);
      bank_out = bank_outs.(t);
      after_wf = after_wf.(t);
      normed2 = normed2.(t);
      rotated;
    })
  ) in
  let outputs = Array.map fst caches in
  let cache = Array.map snd caches in
  (outputs, cache)

(* --- Backward --- *)

(** Wave mix backward: accumulate gradients, return d_normed2 *)
let wave_mix_backward layer cache d_output =
  let n = layer.n_osc in
  let dim = layer.dim in
  let rotated = cache.rotated in
  let normed2 = cache.normed2 in

  (* Through scale → coupling + Duffing → rotation *)
  let d_inner = Array.make dim 0.0 in
  let d_rotated = Array.make dim 0.0 in
  let fold = layer.fold_offset in

  for k = 0 to n - 1 do
    let partner = (k + fold) mod n in
    for off = 0 to 1 do
      let i = k + off * n and pi = partner + off * n in
      let ri = rotated.(i) in
      let cubic = ri +. layer.epsilon.(k) *. ri *. ri *. ri in
      let inner = cubic +. layer.coupling.(k) *. rotated.(pi) in
      (* Scale *)
      d_inner.(i) <- d_output.(i) *. layer.scale.(k);
      layer.d_scale.(k) <- layer.d_scale.(k) +. d_output.(i) *. inner;
      (* Duffing *)
      d_rotated.(i) <- d_rotated.(i)
        +. d_inner.(i) *. (1.0 +. 3.0 *. layer.epsilon.(k) *. ri *. ri);
      layer.d_epsilon.(k) <- layer.d_epsilon.(k) +. d_inner.(i) *. ri *. ri *. ri;
      (* Butterfly fold *)
      d_rotated.(pi) <- d_rotated.(pi) +. d_inner.(i) *. layer.coupling.(k);
      layer.d_coupling.(k) <- layer.d_coupling.(k) +. d_inner.(i) *. rotated.(pi)
    done
  done;

  (* Through rotation *)
  let d_normed2 = Array.make dim 0.0 in
  for k = 0 to n - 1 do
    let c = cos layer.rotation.(k) and s = sin layer.rotation.(k) in
    let dr_pos = d_rotated.(k) and dr_vel = d_rotated.(k + n) in
    d_normed2.(k) <- dr_pos *. c +. dr_vel *. s;
    d_normed2.(k + n) <- -. dr_pos *. s +. dr_vel *. c;
    let xk = normed2.(k) and vk = normed2.(k + n) in
    layer.d_rotation.(k) <- layer.d_rotation.(k)
      +. dr_pos *. (-. xk *. s -. vk *. c)
      +. dr_vel *. (xk *. c -. vk *. s)
  done;
  d_normed2

(** Interference gate backward: accumulate ref_phase gradients, return d_normed1 *)
let interference_gate_backward layer cache d_gated =
  let n_osc = layer.n_osc in
  let dim = layer.dim in
  let d_alignment = Array.make n_osc 0.0 in
  for i = 0 to dim - 1 do
    let k = i mod n_osc in
    d_alignment.(k) <- d_alignment.(k) +. d_gated.(i) *. cache.normed1.(i)
  done;
  for k = 0 to n_osc - 1 do
    let sin_diff = sin (cache.phases.(k) -. layer.ref_phases.(k)) in
    layer.d_ref_phases.(k) <- layer.d_ref_phases.(k) +. 0.5 *. sin_diff *. d_alignment.(k)
  done;
  let d_normed1 = Array.init dim (fun i ->
    d_gated.(i) *. cache.alignments.(i mod n_osc)
  ) in
  for k = 0 to n_osc - 1 do
    let sin_diff = sin (cache.phases.(k) -. layer.ref_phases.(k)) in
    let d_phase = -0.5 *. sin_diff *. d_alignment.(k) in
    let xk = cache.normed1.(k) and vk = cache.normed1.(k + n_osc) in
    let r2 = xk *. xk +. vk *. vk +. 1e-8 in
    d_normed1.(k) <- d_normed1.(k) -. d_phase *. vk /. r2;
    d_normed1.(k + n_osc) <- d_normed1.(k + n_osc) +. d_phase *. xk /. r2
  done;
  d_normed1

(** RMSNorm backward *)
let rms_norm_backward x d_output =
  let n = Float.of_int (Array.length x) in
  let rms = sqrt (Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x /. n +. 1e-8) in
  let inv_rms = 1.0 /. rms in
  let dot = Array.fold_left (fun acc i ->
    acc +. d_output.(i) *. x.(i)
  ) 0.0 (Array.init (Array.length x) Fun.id) in
  Array.init (Array.length x) (fun i ->
    inv_rms *. (d_output.(i) -. x.(i) *. dot /. (rms *. rms *. n))
  )

(** Energy normalization backward *)
let energy_norm_backward x d_output n_osc =
  let n = Float.of_int n_osc in
  let e = ref 0.0 in
  for k = 0 to n_osc - 1 do
    e := !e +. x.(k) *. x.(k) +. x.(k + n_osc) *. x.(k + n_osc)
  done;
  let rms = sqrt (!e /. n +. 1e-8) in
  let inv_rms = 1.0 /. rms in
  let dot = ref 0.0 in
  for i = 0 to 2 * n_osc - 1 do
    dot := !dot +. d_output.(i) *. x.(i)
  done;
  Array.init (2 * n_osc) (fun i ->
    inv_rms *. (d_output.(i) -. x.(i) *. !dot /. (rms *. rms *. n))
  )

(** Backward through full sequence. FFT adjoint + causal EMA BPTT. *)
let backward_sequence layer (cache : position_cache array) d_outputs =
  let seq_len = Array.length cache in

  (* Per-position: wave mix backward → d_after_wf *)
  let d_after_wf = Array.init seq_len (fun t ->
    let d_normed2 = wave_mix_backward layer cache.(t) d_outputs.(t) in
    Array.map2 ( +. ) d_outputs.(t)
      (energy_norm_backward cache.(t).after_wf d_normed2 layer.n_osc)
  ) in

  (* Full-sequence: FFT adjoint through oscillator bank *)
  let d_causal_drives = Bank.encode_backward layer.kernels d_after_wf in

  (* Backward through causal EMA: BPTT *)
  let n_osc = layer.n_osc in
  let d_drives = Array.init seq_len (fun _ -> Array.make n_osc 0.0) in
  let delta = Array.make seq_len 0.0 in
  for k = 0 to n_osc - 1 do
    let beta = Float.max 0.0 (Float.min 0.95 layer.causal_decay.(k)) in
    delta.(seq_len - 1) <- d_causal_drives.(seq_len - 1).(k);
    for t = seq_len - 2 downto 0 do
      delta.(t) <- d_causal_drives.(t).(k) +. beta *. delta.(t + 1)
    done;
    d_drives.(0).(k) <- delta.(0);
    for t = 1 to seq_len - 1 do
      d_drives.(t).(k) <- (1.0 -. beta) *. delta.(t)
    done;
    let ds_db = ref 0.0 in
    for t = 1 to seq_len - 1 do
      let s_prev = cache.(t - 1).causal_drive.(k) in
      let x_t = cache.(t).gated.(k) in
      ds_db := (s_prev -. x_t) +. beta *. !ds_db;
      layer.d_causal_decay.(k) <- layer.d_causal_decay.(k) +. delta.(t) *. !ds_db
    done
  done;

  (* Per-position: interference gate backward → d_input *)
  Array.init seq_len (fun t ->
    let d_gated = Array.init layer.dim (fun i ->
      if i < n_osc then d_drives.(t).(i) else 0.0
    ) in
    let d_normed1 = interference_gate_backward layer cache.(t) d_gated in
    clip_norm 5.0 (Array.map2 ( +. ) d_after_wf.(t)
      (energy_norm_backward cache.(t).input d_normed1 layer.n_osc))
  )

(** Apply accumulated gradients. Weight decay on gate/coupling params only. *)
let apply_gradients layer ~lr ~batch_size =
  let s = -. lr /. Float.of_int batch_size in
  let decay = 0.999 in
  let step_decay w dw n =
    for i = 0 to n - 1 do
      w.(i) <- w.(i) *. decay +. s *. dw.(i);
      dw.(i) <- 0.0
    done
  in
  let step w dw n =
    for i = 0 to n - 1 do
      w.(i) <- w.(i) +. s *. dw.(i);
      dw.(i) <- 0.0
    done
  in
  step_decay layer.ref_phases layer.d_ref_phases layer.n_osc;
  step_decay layer.causal_decay layer.d_causal_decay layer.n_osc;
  for i = 0 to layer.n_osc - 1 do
    layer.causal_decay.(i) <- Float.max 0.0 (Float.min 0.95 layer.causal_decay.(i))
  done;
  step layer.epsilon layer.d_epsilon layer.n_osc;
  step_decay layer.coupling layer.d_coupling layer.n_osc;
  step_decay layer.rotation layer.d_rotation layer.n_osc;
  step layer.scale layer.d_scale layer.n_osc
