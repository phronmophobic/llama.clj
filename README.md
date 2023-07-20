# llama.clj

A wrapper for [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Dependency

```clojure
com.phronemophobic/llama-clj {:mvn/version "0.1.0"}
```

### Native Dependency

llama.clj relies on the excellent [llama.cpp](https://github.com/ggerganov/llama.cpp).

The llama.cpp shared library can either be compiled locally or can be included as a standalone maven dependency.

The easiest method is include the corresponding native dependency for your platform (including multiple is fine, but will increase the size of your dependencies).

```clojure
com.phronemophobic.cljonda/llama-cpp-darwin-aarch64 {:mvn/version "e274269fd87aac0f71ab02a2c4676f60fd6198cf"}
com.phronemophobic.cljonda/llama-cpp-darwin-x86-64 {:mvn/version "e274269fd87aac0f71ab02a2c4676f60fd6198cf"}
com.phronemophobic.cljonda/llama-cpp-linux-x86-64 {:mvn/version "e274269fd87aac0f71ab02a2c4676f60fd6198cf"}
```

#### Locally compiled

Clone https://github.com/ggerganov/llama.cpp and follow the instructions for building. Make sure to include the shared library options.

For Example:

```sh
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release
```

Next include an alias that includes the path where the shared library is located:
```clojure
;; in aliases
;; add jvm opt for local llama build.
:local-llama {:jvm-opts ["-Djna.library.path=/path/to/llama.cpp/build/"]}
```

### Obtaining models

_insert helpful docs for obtaining model here_

## Usage

```sh
clojure -M -m com.phronemophobic.llama <path-to-model> <prompt>
```
Example:

```bash
clojure -M:project -m com.phronemophobic.llama "/models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q4_0.bin" "what is 2+2?"
```

## License

The MIT License (MIT)

Copyright © 2023 Adrian Smith

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



