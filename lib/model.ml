(** Resonance — strike and listen, untied.

    tokens |> strike |> resonate |> listen |> softmax

    Strike and listen use SEPARATE signatures. A good force
    for driving oscillator 5 isn't necessarily a good pattern
    for detecting "the letter e" in the output. They specialize.

    Still no matrix multiply. Just dot products both ways.
    The bank IS the model. Learning IS retuning signatures. *)

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
  strike_sig : float array array;  (** 256 × n_osc: how each byte drives the bank *)
  listen_sig : float array array;  (** 256 × state_dim: what pattern to detect *)
  mutable kernels : Bank.kernels;
  n_osc : int;
  state_dim : int;
  seq_len : int;
}

let create n_osc seq_len =
  let state_dim = 2 * n_osc in
  let scale_s = 1.0 /. sqrt (Float.of_int n_osc) in
  let scale_l = 1.0 /. sqrt (Float.of_int state_dim) in
  let oscillators = Oscillator.spread n_osc in
  {
    oscillators;
    strike_sig = Array.init vocab_size (fun _ ->
      Array.init n_osc (fun _ -> (Random.float 2.0 -. 1.0) *. scale_s));
    listen_sig = Array.init vocab_size (fun _ ->
      Array.init state_dim (fun _ -> (Random.float 2.0 -. 1.0) *. scale_l));
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; state_dim; seq_len;
  }

(* --- Pipeline --- *)

(** Strike: byte → force vector *)
let strike model token = model.strike_sig.(token)

(** Resonate: drives → FFT convolve → causal states *)
let resonate model tokens =
  let drives = Array.map (strike model) tokens in
  Bank.encode model.oscillators model.kernels drives

(** Listen: dot product of state with each byte's listen signature *)
let listen model state =
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to model.state_dim - 1 do
      acc := !acc +. state.(i) *. sig_.(i)
    done; !acc
  ) model.listen_sig

(** Forward: state → logits *)
let forward model state = listen model state

(** Predict: state → probabilities *)
let predict model state = forward model state |> softmax

(* --- Learning --- *)

(** Update listen signatures from output gradient *)
let update_listen model state d_logits ~lr =
  Array.iteri (fun b sig_ ->
    let dl = d_logits.(b) in
    if Float.abs dl > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. lr *. dl *. state.(j)
      ) sig_
  ) model.listen_sig

(** Update strike signatures from state error.
    The gradient flows: d_logits → listen → state → bank → strike.
    Approximate: use the listen gradient projected back. *)
let update_strike model d_logits ~state:_ ~target ~lr =
  (* d_state = listen_sig^T × d_logits *)
  let d_state = Array.init model.state_dim (fun j ->
    let acc = ref 0.0 in
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-8 then acc := !acc +. dl *. sig_.(j)
    ) model.listen_sig; !acc) in
  (* Update the current token's strike signature *)
  let sig_ = model.strike_sig.(target) in
  Array.iteri (fun j _ ->
    if j < model.n_osc then
      sig_.(j) <- sig_.(j) -. lr *. d_state.(j)
  ) sig_

(** Learn from one position *)
let learn_position model ~state ~target ~token ~lr =
  let probs = predict model state in
  let loss = cross_entropy ~target probs in
  let d_logits = logit_gradient ~target probs in
  update_listen model state d_logits ~lr;
  update_strike model d_logits ~state ~target:token ~lr;
  loss

(* --- Training --- *)

let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let lr_scaled = lr /. Float.of_int seq_len in
  let states = resonate model tokens in
  let total_loss = ref 0.0 in
  for t = 0 to seq_len - 2 do
    total_loss := !total_loss +.
      learn_position model ~state:states.(t)
        ~target:tokens.(t + 1) ~token:tokens.(t) ~lr:lr_scaled
  done;
  !total_loss /. Float.of_int (seq_len - 1)

(* --- Generation --- *)

let generate model seed ~n_gen ~temperature =
  let buf = Buffer.create n_gen in
  let context = Array.copy seed in
  let n = Array.length context in
  for _ = 1 to n_gen do
    let states = resonate model context in
    let logits = forward model states.(n - 1) in
    let next = sample ~temperature logits in
    Buffer.add_char buf
      (if next >= 32 && next < 127 then Char.chr next else '.');
    Array.blit context 1 context 0 (n - 1);
    context.(n - 1) <- next
  done;
  Buffer.contents buf
