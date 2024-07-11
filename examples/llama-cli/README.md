# llama-cli

Example usage of llama.clj with native image.

## Native image compilation

```bash
clojure -T:build uber
./compile_native.sh
```

Usage:

```bash
./target/llama-cli <model-path> <prompt>
```

## License

Copyright © 2024 Adrian

Distributed under the Eclipse Public License version 1.0.
