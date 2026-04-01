(** Prism: structured transform through composed oscillator layers.

    Each layer: element-wise scale and shift — each frequency independently.
    Between layers: bit-reversal shuffle — routes frequencies to new positions.
    O(log n) layers composed: approximates expressive transforms.

    Like light through a prism: each frequency passes through
    with its own attenuation and phase shift. Physical. Stable.
    No pair mixing — no feedback loops, no gradient explosion. *)

type layer = {
  scale : float array;
  shift : float array;
}

type t = {
  layers : layer array;
  shuffle : int array;
  dim : int;
  n_layers : int;
}

let bit_reverse_perm n =
  let bits = max 1 (int_of_float (ceil (log (Float.of_int n) /. log 2.0))) in
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

(** Apply one layer: scale and shift each frequency *)
let apply_layer layer x =
  Array.mapi (fun i xi -> xi *. layer.scale.(i) +. layer.shift.(i)) x

(** Apply shuffle *)
let apply_shuffle shuffle x =
  Array.init (Array.length x) (fun i -> x.(shuffle.(i)))

(** Forward: layers interleaved with shuffles *)
let forward prism x =
  Array.fold_left (fun state layer ->
    apply_layer layer state |> apply_shuffle prism.shuffle
  ) x prism.layers

(** Backward: gradient through layers, update params *)
let backward prism x d_output ~lr =
  let intermediates = Array.make (prism.n_layers + 1) x in
  let state = ref x in
  for i = 0 to prism.n_layers - 1 do
    state := apply_layer prism.layers.(i) !state |> apply_shuffle prism.shuffle;
    intermediates.(i + 1) <- !state
  done;

  let inv_shuffle = Array.make prism.dim 0 in
  Array.iteri (fun src dst -> inv_shuffle.(dst) <- src) prism.shuffle;

  let d = ref d_output in
  for i = prism.n_layers - 1 downto 0 do
    d := Array.init prism.dim (fun j -> !d.(inv_shuffle.(j)));
    let pre = intermediates.(i) in
    let layer = prism.layers.(i) in
    Array.iteri (fun j dj ->
      layer.scale.(j) <- layer.scale.(j) -. lr *. dj *. pre.(j);
      layer.shift.(j) <- layer.shift.(j) -. lr *. dj
    ) !d;
    d := Array.mapi (fun j dj -> dj *. layer.scale.(j)) !d
  done;
  !d
