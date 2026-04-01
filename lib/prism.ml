(** Prism: structured transform through composed oscillator layers.

    Replaces dense matrix multiply W × state with:
      couple → shuffle → couple → shuffle → ...

    Each couple: element-wise scaling by learned (amplitude, phase) — O(n)
    Each shuffle: fixed bit-reversal permutation — O(n)
    Composed O(log n) times: approximates O(n²) expressivity.

    Like light through a prism: frequencies separate, recombine,
    and the composition creates any spectral transformation. *)

(** One prism layer: scale + phase shift per element *)
type layer = {
  scale : float array;    (** amplitude per dimension *)
  shift : float array;    (** phase shift per dimension *)
}

(** A prism: composed layers with shuffles between them *)
type t = {
  layers : layer array;
  shuffle : int array;    (** fixed permutation *)
  dim : int;
  n_layers : int;
}

(** Bit-reversal permutation — the classic FFT shuffle *)
let bit_reverse_perm n =
  let bits = int_of_float (ceil (log (Float.of_int n) /. log 2.0)) in
  Array.init n (fun i ->
    let rev = ref 0 in
    let v = ref i in
    for _ = 0 to bits - 1 do
      rev := !rev * 2 + !v mod 2;
      v := !v / 2
    done;
    !rev mod n)

let create dim =
  let n_layers = max 2 (int_of_float (ceil (log (Float.of_int dim) /. log 2.0))) in
  let scale = 1.0 /. sqrt (Float.of_int n_layers) in
  {
    layers = Array.init n_layers (fun _ -> {
      scale = Array.init dim (fun _ -> 1.0 +. (Random.float 2.0 -. 1.0) *. scale);
      shift = Array.init dim (fun _ -> (Random.float 2.0 -. 1.0) *. scale);
    });
    shuffle = bit_reverse_perm dim;
    dim;
    n_layers;
  }

(** Apply one layer: scale and shift each element.
    Treating pairs as (pos, vel), this is amplitude + phase modulation. *)
let apply_layer layer x =
  Array.mapi (fun i xi ->
    xi *. layer.scale.(i) +. layer.shift.(i)
  ) x

(** Apply shuffle permutation *)
let apply_shuffle shuffle x =
  Array.init (Array.length x) (fun i -> x.(shuffle.(i)))

(** Forward through the prism: couple → shuffle → couple → ... *)
let forward prism x =
  let state = ref x in
  Array.iter (fun layer ->
    state := apply_layer layer !state;
    state := apply_shuffle prism.shuffle !state
  ) prism.layers;
  !state

(** Backward: compute gradient of loss w.r.t. prism parameters.
    Returns d_input for chaining. *)
let backward prism x d_output ~lr =
  (* We need intermediate states for the backward pass *)
  let intermediates = Array.make (prism.n_layers + 1) x in
  let state = ref x in
  for i = 0 to prism.n_layers - 1 do
    state := apply_layer prism.layers.(i) !state;
    state := apply_shuffle prism.shuffle !state;
    intermediates.(i + 1) <- !state
  done;

  (* Backward through layers in reverse *)
  let d = ref d_output in
  for i = prism.n_layers - 1 downto 0 do
    (* Unshuffle the gradient (inverse permutation) *)
    let inv_shuffle = Array.make prism.dim 0 in
    Array.iteri (fun src dst -> inv_shuffle.(dst) <- src) prism.shuffle;
    d := Array.init prism.dim (fun j -> !d.(inv_shuffle.(j)));

    (* Gradient through the layer *)
    let pre = intermediates.(i) in
    let layer = prism.layers.(i) in
    (* d_loss/d_scale_j = d_j × pre_j *)
    (* d_loss/d_shift_j = d_j *)
    Array.iteri (fun j dj ->
      layer.scale.(j) <- layer.scale.(j) -. lr *. dj *. pre.(j);
      layer.shift.(j) <- layer.shift.(j) -. lr *. dj
    ) !d;

    (* d_loss/d_pre_j = d_j × scale_j *)
    d := Array.mapi (fun j dj -> dj *. layer.scale.(j)) !d
  done;
  !d
