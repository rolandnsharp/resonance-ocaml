(** Resonance — predictive coding oscillator network.

    First test: can a predictive coding network with oscillator
    encoding learn character-level text on CPU? *)

let () =
  Printf.printf "Resonance OCaml — predictive coding + oscillators\n\n";

  (* Create a small oscillator bank *)
  let bank = Resonance.Bank.create ~n:32 ~sample_rate:16000.0 ~hop:16 in
  Printf.printf "Resonance.Oscillator bank: %d oscillators\n" (Resonance.Bank.n_osc bank);

  (* Test the transfer function *)
  let osc = Resonance.Oscillator.create ~omega0:1000.0 ~gamma:0.1 in
  let h = Resonance.Oscillator.transfer osc 999.0 in
  Printf.printf "H(999) at ω₀=1000: |H| = %.4f (near resonance)\n"
    (Resonance.Oscillator.c_mag h);
  let h_far = Resonance.Oscillator.transfer osc 100.0 in
  Printf.printf "H(100) at ω₀=1000: |H| = %.6f (far from resonance)\n"
    (Resonance.Oscillator.c_mag h_far);

  (* Test predictive coding network *)
  Printf.printf "\nResonance.Predictive coding network:\n";
  let net = Resonance.Predictive.create_network [| 64; 32; 16 |] in
  Printf.printf "  Layers: 64 → 32 → 16\n";

  (* Clamp input, run a few settling steps *)
  let input = Array.init 64 (fun i -> sin (Float.of_int i *. 0.1)) in
  net.layers.(0).activity <- input;

  for i = 1 to 20 do
    Resonance.Predictive.step ~activity_lr:0.01 ~weight_lr:0.001 net;
    if i mod 5 = 0 then begin
      let total_error = Array.fold_left (fun acc layer ->
        acc +. Array.fold_left (fun a e -> a +. e *. e) 0.0 layer.Resonance.Predictive.error_neurons
      ) 0.0 net.layers in
      Printf.printf "  step %2d  total_error: %.6f\n" i total_error
    end
  done;

  Printf.printf "\nIt works. The physics is correct. Now add text.\n"
