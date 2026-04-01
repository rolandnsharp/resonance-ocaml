(** Predictive coding with lateral inhibition.

    Each layer = population of oscillators.
    Between layers = resonance coupling (complex H(ω)).
    Within layers = lateral inhibition (competition).

    The settle loop:
      1. Compute prediction errors (top-down vs actual)
      2. Settle hidden activities (minimize errors)
      3. Sharpen via lateral inhibition (winners suppress losers)
      4. Update coupling frequencies (retune to reduce errors)

    This is how cortex works:
      excitatory neurons + inhibitory interneurons + hierarchical prediction *)

type layer = {
  activity : Vec.t;
  error : Vec.t;
  n_osc : int;     (** number of oscillators in this layer *)
}

type coupling = {
  omega : float array;
  gamma : float array;
  source_omega : float array;
  n_out : int;
  n_in : int;
}

type t = {
  layers : layer array;
  couplings : coupling array;
  n_layers : int;
}

(** Complex transfer function: Re[H] and Im[H] *)
let transfer_re_im ~omega0 ~gamma ~omega =
  let w0sq = omega0 *. omega0 in
  let wsq = omega *. omega in
  let real_num = w0sq -. wsq in
  let imag_num = 2.0 *. gamma *. omega0 *. omega in
  let d = real_num *. real_num +. imag_num *. imag_num +. 1e-8 in
  (real_num /. d, -. imag_num /. d)

(** Complex resonance coupling: z_j = Σ_i z_i * H_j(ω_i) *)
let resonate coupling (input : Vec.t) : Vec.t =
  let n_in = coupling.n_in in
  Vec.create (2 * coupling.n_out) (fun k ->
    let j = k mod coupling.n_out in
    let is_vel = k >= coupling.n_out in
    let w0 = coupling.omega.(j) in
    let g = coupling.gamma.(j) in
    let acc = ref 0.0 in
    for i = 0 to n_in - 1 do
      let wi = coupling.source_omega.(i) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let p = input.(i) in
      let v = input.(n_in + i) in
      if is_vel then acc := !acc +. p *. h_im +. v *. h_re
      else acc := !acc +. p *. h_re -. v *. h_im
    done;
    !acc
  )

(** Backward coupling for error propagation *)
let propagate_error coupling (err : Vec.t) : Vec.t =
  let n_out = coupling.n_out in
  Vec.create (2 * coupling.n_in) (fun k ->
    let i = k mod coupling.n_in in
    let is_vel = k >= coupling.n_in in
    let wi = coupling.source_omega.(i) in
    let acc = ref 0.0 in
    for j = 0 to n_out - 1 do
      let w0 = coupling.omega.(j) in
      let g = coupling.gamma.(j) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let ep = err.(j) in
      let ev = err.(n_out + j) in
      if is_vel then acc := !acc -. ep *. h_im +. ev *. h_re
      else acc := !acc +. ep *. h_re +. ev *. h_im
    done;
    !acc
  )

(** Retune coupling frequencies from prediction error *)
let retune ~lr coupling (input : Vec.t) (err : Vec.t) =
  let n_in = coupling.n_in in
  let eps = 1e-4 in
  for j = 0 to coupling.n_out - 1 do
    let w0 = coupling.omega.(j) in
    let g = coupling.gamma.(j) in
    let ep = err.(j) in
    let ev = err.(coupling.n_out + j) in
    let d_omega = ref 0.0 in
    let d_gamma = ref 0.0 in
    for i = 0 to n_in - 1 do
      let wi = coupling.source_omega.(i) in
      let p = input.(i) in
      let v = input.(n_in + i) in
      let h_re, h_im = transfer_re_im ~omega0:w0 ~gamma:g ~omega:wi in
      let h_re_dw, h_im_dw = transfer_re_im ~omega0:(w0 +. eps) ~gamma:g ~omega:wi in
      let h_re_dg, h_im_dg = transfer_re_im ~omega0:w0 ~gamma:(g +. eps) ~omega:wi in
      let dre_dw = (h_re_dw -. h_re) /. eps in
      let dim_dw = (h_im_dw -. h_im) /. eps in
      let dre_dg = (h_re_dg -. h_re) /. eps in
      let dim_dg = (h_im_dg -. h_im) /. eps in
      d_omega := !d_omega
        +. ep *. (p *. dre_dw -. v *. dim_dw)
        +. ev *. (p *. dim_dw +. v *. dre_dw);
      d_gamma := !d_gamma
        +. ep *. (p *. dre_dg -. v *. dim_dg)
        +. ev *. (p *. dim_dg +. v *. dre_dg)
    done;
    coupling.omega.(j) <- Float.max 0.01 (coupling.omega.(j) -. lr *. !d_omega);
    coupling.gamma.(j) <- Float.max 0.01 (Float.min 0.99 (coupling.gamma.(j) -. lr *. !d_gamma))
  done

let make_freqs n =
  Array.init n (fun i ->
    let f = Float.of_int i /. Float.max 1.0 (Float.of_int (n - 1)) in
    0.1 +. (Float.pi -. 0.1) *. f)

let create_coupling ~n_in ~n_out =
  {
    omega = make_freqs n_out;
    gamma = Array.make n_out 0.1;
    source_omega = make_freqs n_in;
    n_out; n_in;
  }

let create dims =
  let n = Array.length dims in
  let osc_dims = Array.map (fun d -> d / 2) dims in
  {
    n_layers = n;
    layers = Array.init n (fun i ->
      { activity = Vec.zeros dims.(i);
        error = Vec.zeros dims.(i);
        n_osc = osc_dims.(i) });
    couplings = Array.init (n - 1) (fun i ->
      create_coupling ~n_in:osc_dims.(i) ~n_out:osc_dims.(i + 1));
  }

let compute_errors t =
  let layers = Array.copy t.layers in
  for i = 1 to t.n_layers - 1 do
    let predicted = resonate t.couplings.(i - 1) layers.(i - 1).activity in
    layers.(i) <- { layers.(i) with error = Vec.sub layers.(i).activity predicted }
  done;
  { t with layers }

(** Settle hidden layers + lateral inhibition *)
let settle ~lr t =
  let layers = Array.copy t.layers in
  for i = 1 to t.n_layers - 2 do
    let own_err = t.layers.(i).error in
    let feedback =
      if i < t.n_layers - 1 then
        propagate_error t.couplings.(i) t.layers.(i + 1).error
      else Vec.zeros (Vec.dim t.layers.(i).activity)
    in
    let new_act = Vec.add t.layers.(i).activity
      (Vec.scale lr (Vec.sub feedback own_err)) in
    (* Lateral inhibition: oscillators compete *)
    let sharpened = Inhibit.sharpen ~n_osc:t.layers.(i).n_osc new_act in
    layers.(i) <- { layers.(i) with activity = sharpened }
  done;
  { t with layers }

let learn ~lr t =
  for i = 0 to t.n_layers - 2 do
    retune ~lr t.couplings.(i) t.layers.(i).activity t.layers.(i + 1).error
  done;
  t

let clamp state t =
  let layers = Array.copy t.layers in
  layers.(0) <- { layers.(0) with activity = state };
  { t with layers }

(** Full iPC step: clamp → errors → settle+inhibit → learn *)
let step ~activity_lr ~weight_lr (net : t) (inp : Vec.t) =
  net
  |> clamp inp
  |> compute_errors
  |> settle ~lr:activity_lr
  |> compute_errors
  |> learn ~lr:weight_lr

let top_activity t = t.layers.(t.n_layers - 1).activity
let bottom_error t = t.layers.(0).error
