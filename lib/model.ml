(** Deep Resonance — parametric oscillator architecture.

    bank(FFT, once) → [parametric_oscillator → W → SineGate → residual] × L → listen

    Each layer's core is one equation:
    ẍ + 2γ(x_t)ω₀ẋ + ω₀²x = β(x_t)·drive(t), read through c(x_t)

    γ(t): input-dependent damping (how fast to forget)
    β(t): input-dependent amplitude (how strongly to absorb)
    c(t): input-dependent sensitivity (how much to read)

    All three computed from current state. All physically motivated.
    Online updates at every position for fast adaptation. *)

let vocab_size = 256

(* --- Math --- *)

let softmax v =
  let mx = Array.fold_left Float.max neg_infinity v in
  let exps = Array.map (fun l -> exp (l -. mx)) v in
  let sum = Array.fold_left ( +. ) 0.0 exps in
  Array.map (fun e -> e /. sum) exps

let cross_entropy ~target probs =
  -. log (Float.max 1e-10 probs.(target))

let logit_gradient ~target probs =
  Array.mapi (fun i p -> p -. (if i = target then 1.0 else 0.0)) probs

let sample ~temperature logits =
  let probs = Array.map (fun l -> l /. temperature) logits |> softmax in
  let r = Random.float 1.0 in
  let i = ref 0 and acc = ref 0.0 in
  while !acc < r && !i < Array.length probs - 1 do
    acc := !acc +. probs.(!i); if !acc < r then incr i done; !i

let sigmoid x = 1.0 /. (1.0 +. exp (-. x))

let rms_norm x =
  let n = Float.of_int (Array.length x) in
  let rms = sqrt (Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x /. n +. 1e-8) in
  Array.map (fun xi -> xi /. rms) x

let mat_vec w dim x =
  Array.init dim (fun i ->
    let acc = ref 0.0 in
    let base = i * dim in
    for j = 0 to dim - 1 do
      acc := !acc +. w.(base + j) *. x.(j)
    done; !acc)

let mat_vec_t w dim x =
  Array.init dim (fun j ->
    let acc = ref 0.0 in
    for i = 0 to dim - 1 do
      acc := !acc +. w.(i * dim + j) *. x.(i)
    done; !acc)

(* --- Model --- *)

(** One processing layer: parametric oscillator + spectral recombination *)
type layer = {
  (* Parametric oscillator projections: dim → n_osc *)
  w_gamma : float array;   (** damping: how fast to forget *)
  w_beta : float array;    (** amplitude: how strongly to absorb *)
  w_sense : float array;   (** sensitivity: how much to read *)
  (* Spectral recombination *)
  w_mix : float array;     (** dim × dim *)
  (* Oscillator state (mutable, reset per sequence) *)
  mutable osc_pos : float array;  (** position state *)
  mutable osc_vel : float array;  (** velocity state *)
  n_osc : int;
  dim : int;
}

type t = {
  oscillators : Oscillator.t array;
  drive : float array array;     (** 256 × dim *)
  layers : layer array;
  w_out : float array;           (** dim × dim: final synthesis *)
  mutable kernels : Bank.kernels;
  n_osc : int;
  dim : int;
  n_layers : int;
  seq_len : int;
}

let create ~n_osc ~n_layers ~seq_len =
  let dim = 2 * n_osc in
  let scale = 1.0 /. sqrt (Float.of_int dim) in
  let scale_proj = 1.0 /. sqrt (Float.of_int n_osc) in
  let oscillators = Oscillator.spread n_osc in
  {
    oscillators;
    drive = Array.init vocab_size (fun _ ->
      Array.init dim (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    layers = Array.init n_layers (fun _ -> {
      w_gamma = Array.init (dim * n_osc) (fun _ -> (Random.float 2.0 -. 1.0) *. scale_proj);
      w_beta = Array.init (dim * n_osc) (fun _ -> (Random.float 2.0 -. 1.0) *. scale_proj);
      w_sense = Array.init (dim * n_osc) (fun _ -> (Random.float 2.0 -. 1.0) *. scale_proj);
      w_mix = Array.init (dim * dim) (fun _ -> (Random.float 2.0 -. 1.0) *. scale);
      osc_pos = Array.make n_osc 0.0;
      osc_vel = Array.make n_osc 0.0;
      n_osc; dim;
    });
    w_out = Array.init (dim * dim) (fun _ -> (Random.float 2.0 -. 1.0) *. scale);
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; dim; n_layers; seq_len;
  }

(* --- Pipeline --- *)

let strike model token =
  Array.init model.n_osc (fun i -> model.drive.(token).(i))

let resonate model tokens =
  let drives = Array.map (strike model) tokens in
  Bank.encode model.oscillators model.kernels drives

let listen model state =
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to model.dim - 1 do
      acc := !acc +. state.(i) *. sig_.(i)
    done; !acc
  ) model.drive

(** Project state → n_osc values via learned projection *)
let project w dim n_osc state =
  Array.init n_osc (fun k ->
    let acc = ref 0.0 in
    let base = k * dim in
    for j = 0 to dim - 1 do
      acc := !acc +. w.(base + j) *. state.(j)
    done; sigmoid !acc)

(** One parametric oscillator step per oscillator.
    Returns the sensed output (dim = 2*n_osc). *)
let oscillator_step (ly : layer) ~gamma ~beta ~sense bank_out =
  let n = ly.n_osc in
  for k = 0 to n - 1 do
    let g = gamma.(k) and b = beta.(k) in
    let x = ly.osc_pos.(k) and v = ly.osc_vel.(k) in
    let new_x = g *. x +. (1.0 -. g) *. (b *. bank_out.(k)) in
    let new_v = g *. v +. (1.0 -. g) *. (b *. bank_out.(k + n)) in
    ly.osc_pos.(k) <- new_x;
    ly.osc_vel.(k) <- new_v
  done;
  Array.init ly.dim (fun i ->
    let k = i mod n in
    sense.(k) *. (if i < n then ly.osc_pos.(k) else ly.osc_vel.(k)))

(** Reset oscillator states for a new sequence *)
let reset_oscillators model =
  Array.iter (fun ly ->
    Array.fill ly.osc_pos 0 ly.n_osc 0.0;
    Array.fill ly.osc_vel 0 ly.n_osc 0.0
  ) model.layers

(* --- Online training --- *)

let train_online model tokens ~lr =
  let seq_len = Array.length tokens in
  let lr_pos = lr /. Float.of_int seq_len in
  let bank_states = resonate model tokens in
  let dim = model.dim in
  let n_osc = model.n_osc in
  let total_loss = ref 0.0 in

  reset_oscillators model;

  for t = 0 to seq_len - 2 do
    let bank_t = bank_states.(t) in

    (* Forward through layers *)
    let state = ref (rms_norm bank_t) in
    let layer_data = Array.init model.n_layers (fun l ->
      let ly = model.layers.(l) in
      let normed = rms_norm !state in
      let gamma = project ly.w_gamma dim n_osc normed in
      let beta = project ly.w_beta dim n_osc normed in
      let sense = project ly.w_sense dim n_osc normed in
      let prev_pos = Array.copy ly.osc_pos in
      let prev_vel = Array.copy ly.osc_vel in
      let osc_out = oscillator_step ly ~gamma ~beta ~sense bank_t in
      let mixed = mat_vec ly.w_mix dim osc_out in
      let activated = Array.map (fun x -> x *. sin x) mixed in
      state := Array.map2 ( +. ) !state activated;
      (normed, gamma, beta, sense, prev_pos, prev_vel, osc_out, mixed)
    ) in

    (* Final output *)
    let final = rms_norm !state in
    let out = mat_vec model.w_out dim final in
    let logits = listen model out in
    let probs = softmax logits in
    let target = tokens.(t + 1) in
    total_loss := !total_loss +. cross_entropy ~target probs;

    (* === Backward === *)
    let d_logits = logit_gradient ~target probs in

    (* d_out from listen *)
    let d_out = Array.init dim (fun j ->
      let acc = ref 0.0 in
      Array.iteri (fun b sig_ ->
        let dl = d_logits.(b) in
        if Float.abs dl > 1e-8 then acc := !acc +. dl *. sig_.(j)
      ) model.drive; !acc) in

    (* Update drive *)
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-6 then
        Array.iteri (fun j oj -> sig_.(j) <- sig_.(j) -. lr_pos *. dl *. oj) out
    ) model.drive;

    (* Update w_out *)
    for i = 0 to dim - 1 do
      let d = d_out.(i) in
      if Float.abs d > 1e-8 then begin
        let base = i * dim in
        for j = 0 to dim - 1 do
          model.w_out.(base + j) <- model.w_out.(base + j) -. lr_pos *. d *. final.(j)
        done end
    done;

    (* d_state backward through layers *)
    let d_state = ref (mat_vec_t model.w_out dim d_out) in

    for l = model.n_layers - 1 downto 0 do
      let ly = model.layers.(l) in
      let (normed, gamma, beta, sense, _prev_pos, prev_vel, osc_out, mixed) = layer_data.(l) in

      (* Through residual *)
      let d_activated = !d_state in

      (* Through SineGate: d(x*sin(x))/dx = sin(x) + x*cos(x) *)
      let d_mixed = Array.init dim (fun i ->
        d_activated.(i) *. (sin mixed.(i) +. mixed.(i) *. cos mixed.(i))) in

      (* Update w_mix *)
      for i = 0 to dim - 1 do
        let d = d_mixed.(i) in
        if Float.abs d > 1e-8 then begin
          let base = i * dim in
          for j = 0 to dim - 1 do
            ly.w_mix.(base + j) <- ly.w_mix.(base + j) -. lr_pos *. d *. osc_out.(j)
          done end
      done;

      (* d_osc_out = w_mix^T × d_mixed *)
      let d_osc_out = mat_vec_t ly.w_mix dim d_mixed in

      (* Through sensitivity: osc_out = sense * osc_state *)
      let d_sense = Array.make n_osc 0.0 in
      let d_osc_state = Array.make dim 0.0 in
      for i = 0 to dim - 1 do
        let k = i mod n_osc in
        let osc_val = if i < n_osc then ly.osc_pos.(k) else ly.osc_vel.(k) in
        d_sense.(k) <- d_sense.(k) +. d_osc_out.(i) *. osc_val;
        d_osc_state.(i) <- d_osc_out.(i) *. sense.(k)
      done;

      (* Through oscillator dynamics — approximate: treat as linear for gradients *)
      (* d_gamma, d_beta from the oscillator step *)
      let d_gamma = Array.make n_osc 0.0 in
      let d_beta = Array.make n_osc 0.0 in
      for k = 0 to n_osc - 1 do
        let b = beta.(k) and g = gamma.(k) in
        (* d_gamma: x_new = g*x_prev + (1-g)*b*bank → d/dg = x_prev - b*bank *)
        d_gamma.(k) <- d_osc_state.(k) *. (prev_vel.(k) -. b *. bank_t.(k))
                     +. d_osc_state.(k + n_osc) *. (prev_vel.(k) -. b *. bank_t.(k + n_osc));
        (* d_beta: d/db = (1-g)*bank *)
        d_beta.(k) <- d_osc_state.(k) *. ((1.0 -. g) *. bank_t.(k))
                   +. d_osc_state.(k + n_osc) *. ((1.0 -. g) *. bank_t.(k + n_osc))
      done;

      (* Update projections: w_gamma, w_beta, w_sense *)
      let update_proj w d_param =
        for k = 0 to n_osc - 1 do
          let dp = d_param.(k) *. gamma.(k) *. (1.0 -. gamma.(k)) in (* sigmoid deriv *)
          if Float.abs dp > 1e-8 then begin
            let base = k * dim in
            for j = 0 to dim - 1 do
              w.(base + j) <- w.(base + j) -. lr_pos *. dp *. normed.(j)
            done end
        done
      in
      update_proj ly.w_gamma d_gamma;
      let update_proj_generic w d_param gate =
        for k = 0 to n_osc - 1 do
          let dp = d_param.(k) *. gate.(k) *. (1.0 -. gate.(k)) in
          if Float.abs dp > 1e-8 then begin
            let base = k * dim in
            for j = 0 to dim - 1 do
              w.(base + j) <- w.(base + j) -. lr_pos *. dp *. normed.(j)
            done end
        done
      in
      update_proj_generic ly.w_beta d_beta beta;
      update_proj_generic ly.w_sense d_sense sense;

      (* d_state passes through residual *)
      d_state := !d_state
    done
  done;

  (* Weight decay *)
  let decay = 0.9999 in
  Array.iter (fun ly ->
    Array.iteri (fun i v -> ly.w_gamma.(i) <- v *. decay) ly.w_gamma;
    Array.iteri (fun i v -> ly.w_beta.(i) <- v *. decay) ly.w_beta;
    Array.iteri (fun i v -> ly.w_sense.(i) <- v *. decay) ly.w_sense;
    Array.iteri (fun i v -> ly.w_mix.(i) <- v *. decay) ly.w_mix
  ) model.layers;
  Array.iteri (fun i v -> model.w_out.(i) <- v *. decay) model.w_out;

  !total_loss /. Float.of_int (seq_len - 1)

(* --- Generation --- *)

let generate model seed ~n_gen ~temperature =
  let buf = Buffer.create n_gen in
  let context = Array.copy seed in
  let n = Array.length context in
  for _ = 1 to n_gen do
    reset_oscillators model;
    let bank = resonate model context in
    (* Run through all positions to build oscillator state *)
    for t = 0 to n - 1 do
      let state = ref (rms_norm bank.(t)) in
      Array.iter (fun ly ->
        let normed = rms_norm !state in
        let gamma = project ly.w_gamma ly.dim ly.n_osc normed in
        let beta = project ly.w_beta ly.dim ly.n_osc normed in
        let sense = project ly.w_sense ly.dim ly.n_osc normed in
        let osc_out = oscillator_step ly ~gamma ~beta ~sense bank.(t) in
        let mixed = mat_vec ly.w_mix ly.dim osc_out in
        let activated = Array.map (fun x -> x *. sin x) mixed in
        state := Array.map2 ( +. ) !state activated
      ) model.layers;
      if t = n - 1 then begin
        let final = rms_norm !state in
        let out = mat_vec model.w_out model.dim final in
        let logits = listen model out in
        let next = sample ~temperature logits in
        Buffer.add_char buf
          (if next >= 32 && next < 127 then Char.chr next else '.');
        Array.blit context 1 context 0 (n - 1);
        context.(n - 1) <- next
      end
    done
  done;
  Buffer.contents buf
