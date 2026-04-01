(** Resonance — bytes strike bells, predictive coding learns.

    No backprop. No GPU. No tokenizer.
    Each byte is a hammer. Each oscillator is a bell.
    The ringing is the understanding. *)

let read_file path =
  let ic = open_in path in
  let n = in_channel_length ic in
  let s = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  Bytes.to_string s

let () =
  Random.self_init ();
  Printf.printf "Resonance — bytes strike bells\n\n";

  let text_path = if Array.length Sys.argv > 1 then Sys.argv.(1)
    else "data/shakespeare.txt" in
  let text =
    if Sys.file_exists text_path then read_file text_path
    else (Printf.printf "Using built-in text\n";
      "To be, or not to be, that is the question: \
       Whether tis nobler in the mind to suffer \
       The slings and arrows of outrageous fortune.")
  in
  Printf.printf "Text: %d bytes\n" (String.length text);

  let n_osc = int_of_string (try Sys.getenv "N_OSC" with _ -> "32") in
  let n_layers = int_of_string (try Sys.getenv "N_LAYERS" with _ -> "2") in
  let model = Resonance.Text_model.create ~n_osc ~n_layers in
  Printf.printf "Bank: %d oscillators, %d PC layers\n" n_osc n_layers;
  Printf.printf "No backprop. Local errors only.\n\n";

  let n_steps = 10000 in
  let seq_len = 64 in
  let settle = 5 in
  let lr = 0.001 in
  let text_len = String.length text in

  for step = 0 to n_steps - 1 do
    let start = Random.int (text_len - seq_len - 1) in
    Resonance.Text_model.reset model;

    let total_loss = ref 0.0 in
    let count = ref 0 in

    for i = 0 to seq_len - 2 do
      let token = Char.code text.[start + i] in
      let target = Char.code text.[start + i + 1] in

      let logits = Resonance.Text_model.forward model token ~settle_steps:settle in
      total_loss := !total_loss +. Resonance.Vec.cross_entropy ~target (Resonance.Vec.softmax logits);
      incr count;

      Resonance.Text_model.learn model ~target ~logits ~lr
    done;

    if step mod 200 = 0 then begin
      let avg = !total_loss /. Float.of_int !count in
      let bpc = avg /. log 2.0 in
      Printf.printf "step %5d  loss %.3f  bpc %.3f" step avg bpc;

      if step mod 1000 = 0 then begin
        Resonance.Text_model.reset model;
        let seed = "The " in
        String.iter (fun c ->
          ignore (Resonance.Text_model.forward model (Char.code c) ~settle_steps:settle)
        ) seed;
        Printf.printf "  | The ";
        let last = ref (Char.code 'e') in
        for _ = 1 to 80 do
          let logits = Resonance.Text_model.forward model !last ~settle_steps:settle in
          let next = Resonance.Vec.sample ~temperature:0.8 logits in
          last := next;
          let c = if next >= 32 && next < 127 then Char.chr next else '.' in
          Printf.printf "%c" c;
        done;
      end;
      Printf.printf "\n%!"
    end
  done;

  Printf.printf "\nDone.\n"
