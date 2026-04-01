(** One layer: wave-field attention + FFN.

    Like Kecro88's WaveFieldTransformerLayer:
      norm → wave_field → residual → norm → ffn → residual

    Wave field: content-dependent deposit → FFT convolve → gated gather
    FFN: Linear → GELU → Linear (small at d=256: 65K ops) *)

(** RMSNorm *)
let rms_norm x =
  let n = Float.of_int (Array.length x) in
  let rms = sqrt (Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x /. n +. 1e-8) in
  Array.map (fun xi -> xi /. rms) x

(** Mat-vec: flat row-major W (rows × cols) × x *)
let mat_vec w ~rows ~cols x =
  Array.init rows (fun i ->
    let acc = ref 0.0 in
    let base = i * cols in
    for j = 0 to cols - 1 do
      acc := !acc +. w.(base + j) *. x.(j)
    done; !acc)

(** Outer product gradient accumulation *)
let outer_accum grad a b ~rows:_ ~cols =
  Array.iteri (fun i ai ->
    if Float.abs ai > 1e-8 then
      let base = i * cols in
      Array.iteri (fun j bj ->
        grad.(base + j) <- grad.(base + j) +. ai *. bj
      ) b
  ) a

type t = {
  dim : int;
  hidden_dim : int;
  (* Wave field: deposit strength + gate *)
  w_value : float array;    (** dim × dim: what to deposit *)
  w_key : float array;      (** dim × dim: deposit strength *)
  w_gate : float array;     (** dim × dim: reception gate *)
  (* FFN *)
  w_up : float array;       (** hidden_dim × dim: expand *)
  w_down : float array;     (** dim × hidden_dim: contract *)
  (* Oscillator kernel for wave propagation *)
  oscillators : Oscillator.t array;
  mutable kernels : Bank.kernels;
  n_osc : int;
}

let create ~dim ~n_osc ~seq_len =
  let hidden_dim = dim * 2 in
  let scale = 1.0 /. sqrt (Float.of_int dim) in
  let rand () = (Random.float 2.0 -. 1.0) *. scale in
  let id_init size =
    Array.init (size * size) (fun k ->
      let i = k / size and j = k mod size in
      (if i = j then 1.0 else 0.0) +. rand () *. 0.01)
  in
  let oscillators = Oscillator.spread n_osc in
  {
    dim; hidden_dim;
    w_value = id_init dim;
    w_key = Array.init (dim * dim) (fun _ -> rand () *. 0.01);
    w_gate = Array.init (dim * dim) (fun _ -> rand () *. 0.01);
    w_up = Array.init (hidden_dim * dim) (fun _ -> rand ());
    w_down = Array.init (dim * hidden_dim) (fun _ ->
      (Random.float 2.0 -. 1.0) /. sqrt (Float.of_int hidden_dim));
    oscillators;
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc;
  }

(** GELU activation *)
let gelu x =
  0.5 *. x *. (1.0 +. tanh (0.7978846 *. (x +. 0.044715 *. x *. x *. x)))

(** Wave field forward on a full sequence.
    For each position: value × key_magnitude → deposit, FFT propagate, gate × gather *)
let wave_field layer states =
  let seq_len = Array.length states in
  let dim = layer.dim in
  let n_osc = layer.n_osc in

  (* Content-dependent: value, key magnitude, gate at each position *)
  let values = Array.map (fun s -> mat_vec layer.w_value ~rows:dim ~cols:dim s) states in
  let key_mags = Array.map (fun s ->
    let k = mat_vec layer.w_key ~rows:dim ~cols:dim s in
    let mag = sqrt (Array.fold_left (fun acc ki -> acc +. ki *. ki) 0.0 k +. 1e-8) in
    mag
  ) states in
  let gates = Array.map (fun s ->
    let pre = mat_vec layer.w_gate ~rows:dim ~cols:dim s in
    Array.map (fun x -> 1.0 /. (1.0 +. exp (-. x))) pre
  ) states in

  (* Deposit: scale values by key magnitude *)
  let deposits = Array.init seq_len (fun t ->
    Array.map (fun v -> v *. key_mags.(t)) values.(t)
  ) in

  (* Propagate: FFT convolve deposits through oscillator kernel.
     Use first n_osc dims as drives for the bank *)
  let drives = Array.map (fun dep ->
    Array.init n_osc (fun i -> dep.(i))
  ) deposits in
  let propagated = Bank.encode layer.oscillators layer.kernels drives in

  (* Gather + gate: element-wise multiply *)
  Array.init seq_len (fun t ->
    Array.init dim (fun i ->
      gates.(t).(i) *. propagated.(t).(i mod (2 * n_osc))
    )
  )

(** FFN forward: up → GELU → down *)
let ffn layer x =
  let h = mat_vec layer.w_up ~rows:layer.hidden_dim ~cols:layer.dim x in
  let activated = Array.map gelu h in
  mat_vec layer.w_down ~rows:layer.dim ~cols:layer.hidden_dim activated

(** Full layer forward on a sequence:
    norm → wave_field → residual → norm → ffn → residual *)
let forward layer states =
  (* Wave field attention *)
  let normed = Array.map rms_norm states in
  let wf_out = wave_field layer normed in
  let after_wf = Array.map2 (fun s wf ->
    Array.map2 ( +. ) s wf  (* residual *)
  ) states wf_out in

  (* FFN *)
  let normed2 = Array.map rms_norm after_wf in
  let ffn_out = Array.map (ffn layer) normed2 in
  Array.map2 (fun s ff ->
    Array.map2 ( +. ) s ff  (* residual *)
  ) after_wf ffn_out
