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
  layers : Layer.t array;          (** stacked wave-field + FFN layers *)
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
    layers = Array.init n_layers (fun _ ->
      Layer.create ~dim:state_dim ~n_osc ~seq_len);
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

(** Forward: full pipeline for one position *)
let forward model processed_state =
  processed_state |> Layer.rms_norm |> listen model

(** Predict *)
let predict model state =
  forward model state |> softmax

(* --- Training (forward + numerical gradient on drive weights) --- *)

let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let lr_scaled = lr /. Float.of_int seq_len in

  (* Forward through bank *)
  let initial_states = resonate model tokens in

  (* Forward through all layers, storing caches *)
  let layer_caches = Array.make model.n_layers [||] in
  let current = ref initial_states in
  Array.iteri (fun i layer ->
    let outputs, cache = Layer.forward layer !current in
    layer_caches.(i) <- cache;
    current := outputs
  ) model.layers;
  let final_states = !current in

  (* Compute loss and output gradients at each position *)
  let d_outputs = Array.init (seq_len - 1) (fun t ->
    let normed = Layer.rms_norm final_states.(t) in
    let logits = listen model normed in
    let probs = softmax logits in
    let target = tokens.(t + 1) in
    let d_logits = logit_gradient ~target probs in
    (* Update drive *)
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-6 then
        Array.iteri (fun j _ ->
          sig_.(j) <- sig_.(j) -. lr_scaled *. dl *. normed.(j)
        ) sig_
    ) model.drive;
    (* d_state from listen backward *)
    Array.init model.state_dim (fun j ->
      let acc = ref 0.0 in
      Array.iteri (fun b sig_ ->
        let dl = d_logits.(b) in
        if Float.abs dl > 1e-8 then
          acc := !acc +. dl *. sig_.(j)
      ) model.drive; !acc)
  ) in

  (* Pad d_outputs to seq_len (last position has no target) *)
  let d_full = Array.init seq_len (fun t ->
    if t < seq_len - 1 then d_outputs.(t)
    else Array.make model.state_dim 0.0
  ) in

  (* Backward through layers in reverse *)
  let d_current = ref d_full in
  for i = model.n_layers - 1 downto 0 do
    let _outputs, d_inputs = Layer.train model.layers.(i)
      (if i = 0 then initial_states
       else let outs, _ = Layer.forward model.layers.(i-1) initial_states in outs)
      !d_current ~lr:lr_scaled in
    d_current := d_inputs;
    ignore _outputs
  done;

  (* Compute total loss *)
  let total_loss = ref 0.0 in
  for t = 0 to seq_len - 2 do
    let normed = Layer.rms_norm final_states.(t) in
    let probs = softmax (listen model normed) in
    total_loss := !total_loss +. cross_entropy ~target:tokens.(t + 1) probs
  done;
  !total_loss /. Float.of_int (seq_len - 1)

(* --- Generation --- *)

let generate model seed ~n_gen ~temperature =
  let buf = Buffer.create n_gen in
  let context = Array.copy seed in
  let n = Array.length context in
  for _ = 1 to n_gen do
    let states = resonate model context in
    let processed = process model states in
    let logits = forward model processed.(n - 1) in
    let next = sample ~temperature logits in
    Buffer.add_char buf
      (if next >= 32 && next < 127 then Char.chr next else '.');
    Array.blit context 1 context 0 (n - 1);
    context.(n - 1) <- next
  done;
  Buffer.contents buf
