(** Vectors and matrices — optimized for the hot path.

    Uses Bigarray with C layout for cache-friendly access
    and unsafe operations where bounds are guaranteed. *)

type t = float array

let create n f = Array.init n f
let zeros n = Array.make n 0.0
let dim = Array.length
let copy = Array.copy

let map = Array.map
let mapi = Array.mapi

let map2 f a b =
  let n = dim a in
  let r = Array.make n 0.0 in
  for i = 0 to n - 1 do
    Array.unsafe_set r i (f (Array.unsafe_get a i) (Array.unsafe_get b i))
  done; r

let add a b =
  let n = dim a in
  let r = Array.make n 0.0 in
  for i = 0 to n - 1 do
    Array.unsafe_set r i (Array.unsafe_get a i +. Array.unsafe_get b i)
  done; r

let sub a b =
  let n = dim a in
  let r = Array.make n 0.0 in
  for i = 0 to n - 1 do
    Array.unsafe_set r i (Array.unsafe_get a i -. Array.unsafe_get b i)
  done; r

let scale s v =
  let n = dim v in
  let r = Array.make n 0.0 in
  for i = 0 to n - 1 do
    Array.unsafe_set r i (s *. Array.unsafe_get v i)
  done; r

let hadamard a b =
  let n = dim a in
  let r = Array.make n 0.0 in
  for i = 0 to n - 1 do
    Array.unsafe_set r i (Array.unsafe_get a i *. Array.unsafe_get b i)
  done; r

let dot a b =
  let n = dim a in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do
    acc := !acc +. Array.unsafe_get a i *. Array.unsafe_get b i
  done; !acc

let sum v =
  let n = dim v in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do acc := !acc +. Array.unsafe_get v i done;
  !acc

let norm_sq v = dot v v

(** Matrix as flat float array for cache locality.
    Row-major: element (i,j) at index i*cols+j *)
type mat = {
  data : float array;
  rows : int;
  cols : int;
}

let mat_create ~rows ~cols f =
  let data = Array.init (rows * cols) (fun k ->
    f (k / cols) (k mod cols)
  ) in
  { data; rows; cols }

let mat_rand ~rows ~cols ~scale =
  mat_create ~rows ~cols (fun _ _ -> (Random.float 2.0 -. 1.0) *. scale)

(** Matrix-vector: y = M * x. Hot path — fully unrolled. *)
let mat_vec m x =
  let r = Array.make m.rows 0.0 in
  let cols = m.cols in
  let d = m.data in
  for i = 0 to m.rows - 1 do
    let acc = ref 0.0 in
    let base = i * cols in
    for j = 0 to cols - 1 do
      acc := !acc +. Array.unsafe_get d (base + j) *. Array.unsafe_get x j
    done;
    Array.unsafe_set r i !acc
  done; r

(** Transposed matrix-vector: y = M^T * x *)
let mat_t_vec m x =
  let r = Array.make m.cols 0.0 in
  let cols = m.cols in
  let d = m.data in
  for i = 0 to m.rows - 1 do
    let xi = Array.unsafe_get x i in
    let base = i * cols in
    for j = 0 to cols - 1 do
      Array.unsafe_set r j
        (Array.unsafe_get r j +. Array.unsafe_get d (base + j) *. xi)
    done
  done; r

(** Outer product update: M += lr * a (outer) b *)
let mat_outer_update ~lr m a b =
  let cols = m.cols in
  let d = m.data in
  for i = 0 to m.rows - 1 do
    let ai = lr *. Array.unsafe_get a i in
    let base = i * cols in
    for j = 0 to cols - 1 do
      let idx = base + j in
      Array.unsafe_set d idx
        (Array.unsafe_get d idx +. ai *. Array.unsafe_get b j)
    done
  done

(** Softmax *)
let softmax v =
  let n = dim v in
  let mx = ref neg_infinity in
  for i = 0 to n - 1 do
    let vi = Array.unsafe_get v i in
    if vi > !mx then mx := vi
  done;
  let m = !mx in
  let r = Array.make n 0.0 in
  let s = ref 0.0 in
  for i = 0 to n - 1 do
    let e = exp (Array.unsafe_get v i -. m) in
    Array.unsafe_set r i e;
    s := !s +. e
  done;
  let inv_s = 1.0 /. !s in
  for i = 0 to n - 1 do
    Array.unsafe_set r i (Array.unsafe_get r i *. inv_s)
  done; r

(** Cross-entropy loss *)
let cross_entropy ~target probs =
  -. log (Float.max 1e-10 (Array.unsafe_get probs target))

(** Sample from probability distribution *)
let sample ?(temperature=1.0) v =
  let scaled = map (fun x -> x /. temperature) v in
  let probs = softmax scaled in
  let r = Random.float 1.0 in
  let n = dim probs in
  let rec pick i acc =
    if i >= n - 1 then i
    else let acc' = acc +. Array.unsafe_get probs i in
      if r < acc' then i else pick (i + 1) acc'
  in pick 0 0.0
