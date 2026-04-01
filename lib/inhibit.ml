(** Lateral inhibition — oscillators compete within a layer.

    The brain's cortical columns have excitatory neurons (oscillators)
    and inhibitory interneurons that create winner-take-all dynamics.
    Strong responders suppress weak ones. What survives is content-dependent.

    Divisive normalization (Carandini & Heeger, 2012):
      gate_i = energy_i / (energy_i + mean_energy + σ)

    Each oscillator's energy = pos² + vel².
    Above-average energy → amplified. Below-average → suppressed.
    The result is a sharp, sparse, content-dependent representation. *)

(** Sharpen oscillator state via lateral inhibition.
    state = [pos_0..pos_n, vel_0..vel_n]
    Each oscillator competes based on its energy.
    Both pos and vel are scaled by the same gate (preserves phase). *)
let sharpen ~n_osc (state : Vec.t) : Vec.t =
  (* Energy per oscillator: E_i = pos_i² + vel_i² *)
  let energies = Array.init n_osc (fun i ->
    let p = state.(i) in
    let v = state.(n_osc + i) in
    p *. p +. v *. v
  ) in
  let mean_e = Array.fold_left ( +. ) 0.0 energies /. Float.of_int n_osc in
  (* Divisive normalization gate *)
  Vec.create (2 * n_osc) (fun k ->
    let i = k mod n_osc in
    let gate = energies.(i) /. (energies.(i) +. mean_e +. 1e-8) in
    state.(k) *. gate
  )
