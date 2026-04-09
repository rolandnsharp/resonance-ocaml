(** Resonance — the full architecture.

    tokens
    |> strike (drive lookup)
    |> resonate (initial FFT bank)
    |> layer₁ (wave_field + ffn)
    |> layer₂
    |> ...
    |> layer₆
    |> norm → listen → softmax

    Each layer: wave-field attention (O(n log n)) + FFN (O(d²)).
    At d=256: 65K ops per FFN. CPU-native.
    Scales to 10,000 RISC-V chips. *)

let vocab_size = 256

(* --- Probability --- *)

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

(* --- Model --- *)

type t = {
  oscillators : Oscillator.t array;
  drive : float array array;       (** 256 × state_dim *)
  d_drive : float array array;     (** gradient accumulator *)
  w_select : float array;           (** n_osc × n_osc: selective output gate *)
  d_w_select : float array;
  w : float array;                 (** state_dim × state_dim: synthesis *)
  d_w : float array;
  layers : Layer.t array;
  mutable kernels : Bank.kernels;
  n_osc : int;
  state_dim : int;
  n_layers : int;
  seq_len : int;
}

let create ~n_osc ~n_layers ~seq_len =
  let state_dim = 2 * n_osc in
  let scale = 1.0 /. sqrt (Float.of_int state_dim) in
  let oscillators = Oscillator.spread n_osc in
  {
    oscillators;
    drive = Array.init vocab_size (fun _ ->
      Array.init state_dim (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    d_drive = Array.init vocab_size (fun _ -> Array.make state_dim 0.0);
    w_select = Array.init (n_osc * n_osc) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale);
    d_w_select = Array.make (n_osc * n_osc) 0.0;
    w = Array.init (state_dim * state_dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale);
    d_w = Array.make (state_dim * state_dim) 0.0;
    layers = Array.init n_layers (fun i ->
      Layer.create ~n_osc ~seq_len ~layer_idx:i);
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; state_dim; n_layers; seq_len;
  }

(* --- Pipeline --- *)

(** Strike: first n_osc components of drive *)
let strike model token =
  Array.init model.n_osc (fun i -> model.drive.(token).(i))

(** Resonate: initial FFT bank encoding *)
let resonate model tokens =
  let drives = Array.map (strike model) tokens in
  Bank.encode model.oscillators model.kernels drives

let sigmoid x = 1.0 /. (1.0 +. exp (-. x))

(** Selective gate: current token's drive controls which oscillators to read *)
let select model bank_out token =
  let n = model.n_osc in
  let drive = strike model token in
  let gate = Array.init n (fun k ->
    let acc = ref 0.0 in
    for j = 0 to n - 1 do
      acc := !acc +. model.w_select.(k * n + j) *. drive.(j)
    done; sigmoid !acc
  ) in
  Array.init model.state_dim (fun i ->
    bank_out.(i) *. gate.(i mod n))

(** Process: stack of wave-field + FFN layers (inference only) *)
let process model states =
  Array.fold_left (fun s layer ->
    let outputs, _cache = Layer.forward layer s in outputs
  ) states model.layers

(** Listen: dot product with drive signatures *)
let listen model state =
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to model.state_dim - 1 do
      acc := !acc +. state.(i) *. sig_.(i)
    done; !acc
  ) model.drive

(** Synthesis: W recombines oscillator state for prediction *)
let transform model state =
  let dim = model.state_dim in
  Array.init dim (fun i ->
    let acc = ref 0.0 in
    let base = i * dim in
    for j = 0 to dim - 1 do
      acc := !acc +. model.w.(base + j) *. state.(j)
    done; !acc)

(** Forward: norm → W → listen *)
let forward model processed_state =
  processed_state |> Layer.rms_norm |> transform model |> listen model

(* --- Training: forward, accumulate gradients, apply --- *)

(** Accumulate gradients for one sequence. Returns loss.
    No weights change — call apply_gradients after the batch. *)
let accumulate_gradients model tokens =
  let seq_len = Array.length tokens in
  let inv_t = 1.0 /. Float.of_int (seq_len - 1) in

  (* Forward: strike → resonate → selective gate → layers *)
  let bank_states = resonate model tokens in
  let initial_states = Array.init seq_len (fun t ->
    select model bank_states.(t) tokens.(t)
  ) in
  let layer_caches = Array.make model.n_layers [||] in
  let current = ref initial_states in
  Array.iteri (fun _i layer ->
    let outputs, cache = Layer.forward layer !current in
    layer_caches.(_i) <- cache;
    current := outputs
  ) model.layers;
  let final_states = !current in

  (* Loss + output gradients (computed before any weight updates) *)
  let total_loss = ref 0.0 in
  let d_final = Array.init seq_len (fun t ->
    if t >= seq_len - 1 then Array.make model.state_dim 0.0
    else begin
      let normed = Layer.rms_norm final_states.(t) in
      let transformed = transform model normed in
      let logits = listen model transformed in
      let probs = softmax logits in
      let target = tokens.(t + 1) in
      total_loss := !total_loss +. cross_entropy ~target probs;
      let d_logits = Array.map (fun x -> x *. inv_t)
        (logit_gradient ~target probs) in
      (* Accumulate drive gradients: d_logits ⊗ transformed *)
      Array.iteri (fun b _ ->
        let dl = d_logits.(b) in
        if Float.abs dl > 1e-6 then
          Array.iteri (fun j tj ->
            model.d_drive.(b).(j) <- model.d_drive.(b).(j) +. dl *. tj
          ) transformed
      ) model.drive;
      (* d_transformed from listen: drive^T × d_logits *)
      let dim = model.state_dim in
      let d_transformed = Array.init dim (fun j ->
        let acc = ref 0.0 in
        Array.iteri (fun b sig_ ->
          let dl = d_logits.(b) in
          if Float.abs dl > 1e-8 then
            acc := !acc +. dl *. sig_.(j)
        ) model.drive; !acc
      ) in
      (* W gradients: d_transformed ⊗ normed *)
      for i = 0 to dim - 1 do
        let dt = d_transformed.(i) in
        if Float.abs dt > 1e-8 then begin
          let base = i * dim in
          for j = 0 to dim - 1 do
            model.d_w.(base + j) <- model.d_w.(base + j) +. dt *. normed.(j)
          done end
      done;
      (* Backward: W^T × d_transformed → rms_norm_backward *)
      let d_normed = Array.init dim (fun j ->
        let acc = ref 0.0 in
        for i = 0 to dim - 1 do
          acc := !acc +. model.w.(i * dim + j) *. d_transformed.(i)
        done; !acc) in
      Layer.rms_norm_backward final_states.(t) d_normed
    end
  ) in

  (* Backward through layers in reverse *)
  let d_current = ref d_final in
  for i = model.n_layers - 1 downto 0 do
    d_current := Layer.backward_sequence model.layers.(i)
      layer_caches.(i) !d_current
  done;

  (* Backward through selective gate: d_w_select *)
  let n = model.n_osc in
  for t = 0 to seq_len - 1 do
    let d_gated = !d_current.(t) in
    let drive = strike model tokens.(t) in
    let gate = Array.init n (fun k ->
      let acc = ref 0.0 in
      for j = 0 to n - 1 do
        acc := !acc +. model.w_select.(k * n + j) *. drive.(j)
      done; sigmoid !acc
    ) in
    (* d_w_select: d_gate * sigmoid' * drive *)
    for k = 0 to n - 1 do
      let d_gate_k = ref 0.0 in
      (* gate affects both pos and vel of oscillator k *)
      d_gate_k := !d_gate_k +. d_gated.(k) *. bank_states.(t).(k);
      d_gate_k := !d_gate_k +. d_gated.(k + n) *. bank_states.(t).(k + n);
      let g = gate.(k) in
      let ds = !d_gate_k *. g *. (1.0 -. g) in
      for j = 0 to n - 1 do
        model.d_w_select.(k * n + j) <- model.d_w_select.(k * n + j) +. ds *. drive.(j)
      done
    done
  done;

  !total_loss /. Float.of_int (seq_len - 1)

(** Apply accumulated gradients, reset accumulators. *)
let apply_gradients model ~lr ~batch_size =
  let s = -. lr /. Float.of_int batch_size in
  Array.iteri (fun b sig_ ->
    Array.iteri (fun j _ ->
      sig_.(j) <- sig_.(j) +. s *. model.d_drive.(b).(j);
      model.d_drive.(b).(j) <- 0.0
    ) sig_
  ) model.drive;
  for i = 0 to Array.length model.w_select - 1 do
    model.w_select.(i) <- model.w_select.(i) *. 0.999 +. s *. model.d_w_select.(i);
    model.d_w_select.(i) <- 0.0
  done;
  for i = 0 to Array.length model.w - 1 do
    model.w.(i) <- model.w.(i) *. 0.999 +. s *. model.d_w.(i);
    model.d_w.(i) <- 0.0
  done;
  Array.iter (fun layer ->
    Layer.apply_gradients layer ~lr ~batch_size
  ) model.layers

(** Train one batch: accumulate across sequences, apply once. *)
let train_batch model token_seqs ~lr =
  let batch_size = Array.length token_seqs in
  let total_loss = Array.fold_left (fun acc seq ->
    acc +. accumulate_gradients model seq
  ) 0.0 token_seqs in
  apply_gradients model ~lr ~batch_size;
  total_loss /. Float.of_int batch_size

(** Online training: update W and drive after each position.
    Matches the original architecture's training method. *)
let train_online model tokens ~lr =
  let seq_len = Array.length tokens in
  let lr_pos = lr /. Float.of_int seq_len in
  let bank_states = resonate model tokens in
  let states = Array.init seq_len (fun t ->
    select model bank_states.(t) tokens.(t)) in
  let dim = model.state_dim in
  let total_loss = ref 0.0 in
  for t = 0 to seq_len - 2 do
    let normed = Layer.rms_norm states.(t) in
    let transformed = transform model normed in
    let logits = listen model transformed in
    let probs = softmax logits in
    let target = tokens.(t + 1) in
    total_loss := !total_loss +. cross_entropy ~target probs;
    let d_logits = logit_gradient ~target probs in
    (* Update W immediately: W -= lr * outer(d_transformed, normed) *)
    let d_transformed = Array.init dim (fun j ->
      let acc = ref 0.0 in
      Array.iteri (fun b sig_ ->
        let dl = d_logits.(b) in
        if Float.abs dl > 1e-8 then
          acc := !acc +. dl *. sig_.(j)
      ) model.drive; !acc) in
    for i = 0 to dim - 1 do
      let dt = d_transformed.(i) in
      if Float.abs dt > 1e-8 then begin
        let base = i * dim in
        for j = 0 to dim - 1 do
          model.w.(base + j) <- model.w.(base + j) -. lr_pos *. dt *. normed.(j)
        done end
    done;
    (* Update drive immediately *)
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-6 then
        Array.iteri (fun j tj ->
          sig_.(j) <- sig_.(j) -. lr_pos *. dl *. tj
        ) transformed
    ) model.drive;
    (* Update w_select *)
    let n = model.n_osc in
    let drive = strike model tokens.(t) in
    let gate = Array.init n (fun k ->
      let acc = ref 0.0 in
      for j = 0 to n - 1 do
        acc := !acc +. model.w_select.(k * n + j) *. drive.(j)
      done; sigmoid !acc) in
    for k = 0 to n - 1 do
      let d_gate_k =
        d_transformed.(k) *. bank_states.(t).(k)
        +. d_transformed.(k + n) *. bank_states.(t).(k + n) in
      let g = gate.(k) in
      let ds = d_gate_k *. g *. (1.0 -. g) in
      for j = 0 to n - 1 do
        model.w_select.(k * n + j) <- model.w_select.(k * n + j) -. lr_pos *. ds *. drive.(j)
      done
    done
  done;
  (* Weight decay on W *)
  Array.iteri (fun i v -> model.w.(i) <- v *. 0.9999) model.w;
  !total_loss /. Float.of_int (seq_len - 1)

(* --- Generation --- *)

let generate model seed ~n_gen ~temperature =
  let buf = Buffer.create n_gen in
  let context = Array.copy seed in
  let n = Array.length context in
  for _ = 1 to n_gen do
    let bank = resonate model context in
    let states = Array.init n (fun t -> select model bank.(t) context.(t)) in
    let processed = process model states in
    let logits = forward model processed.(n - 1) in
    let next = sample ~temperature logits in
    Buffer.add_char buf
      (if next >= 32 && next < 127 then Char.chr next else '.');
    Array.blit context 1 context 0 (n - 1);
    context.(n - 1) <- next
  done;
  Buffer.contents buf
