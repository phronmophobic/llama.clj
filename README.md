# llama.clj

Run LLMs locally. A clojure wrapper for [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Quick Start

If you're just looking for a model to try things out, try the 3.6Gb [llama2 7B chat model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)  from TheBloke. Make sure to check the link for important info like license and use policy.

```sh
mkdir -p models
# Download 3.6Gb model to models/ directory
(cd models && curl -L -O 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf')
# mvn-llama alias pulls precompiled llama.cpp libs from maven
clojure -M:mvn-llama -m com.phronemophobic.llama "models/llama-2-7b-chat.Q4_0.gguf" "what is 2+2?"
```

_Note: For best results, prompts should be formatted using the correct prompt format for the particular model being used (eg. [llama2 chat prompt format](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML#prompt-template-llama-2-chat))._

## Documentation

[Getting Started](https://phronmophobic.github.io/llama.clj/)  
[Intro to Running LLMs Locally](https://phronmophobic.github.io/llama.clj/notebooks/intro.html)  
[API Reference Docs](https://phronmophobic.github.io/llama.clj/reference/)  

## Dependency

For llama.clj with required native dependencies:

```clojure
com.phronemophobic/llama-clj-combined {:mvn/version "0.8.1"}
```

For llama.clj only _(see below for various alternatives for specifying native dependencies)_:
```clojure
com.phronemophobic/llama-clj {:mvn/version "0.8.1"}
```

### Native Dependency

llama.clj relies on the excellent [llama.cpp](https://github.com/ggerganov/llama.cpp) library.

The llama.cpp shared library can either be compiled locally or can be included as a standalone maven dependency.

#### Precompiled native deps on clojars

The easiest method is to include the corresponding native dependency for your platform (including multiple is fine, but will increase the size of your dependencies). See the [mvn-llama alias](https://github.com/phronmophobic/llama.clj/blob/b4fef0e8fc23a72349796911cef33d6bbdadcd73/deps.edn#L11) for an example.

```clojure
;; gguf dependencies
com.phronemophobic.cljonda/llama-cpp-gguf-linux-x86-64 {:mvn/version "b2291"}
com.phronemophobic.cljonda/llama-cpp-gguf-darwin-aarch64 {:mvn/version "b2291"}
com.phronemophobic.cljonda/llama-cpp-gguf-darwin-x86-64 {:mvn/version "b2291"}

;; ggml dependencies
com.phronemophobic.cljonda/llama-cpp-darwin-aarch64 {:mvn/version "6e88a462d7d2d281e33f35c3c41df785ef633bc1"}
com.phronemophobic.cljonda/llama-cpp-darwin-x86-64 {:mvn/version "6e88a462d7d2d281e33f35c3c41df785ef633bc1"}
com.phronemophobic.cljonda/llama-cpp-linux-x86-64 {:mvn/version "6e88a462d7d2d281e33f35c3c41df785ef633bc1"}
```

#### Locally compiled

Clone https://github.com/ggerganov/llama.cpp and follow the instructions for building. Make sure to include the shared library options.

_Note: The llama.cpp ffi bindings are based on the `4329d1acb01c353803a54733b8eef9d93d0b84b2` git commit for ggml models and the `b2291` release for gguf models. Future versions of llama.cpp might not be compatible if breaking changes are made. TODO: include instructions for updating ffi bindings._

For Example:

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 4329d1acb01c353803a54733b8eef9d93d0b84b2
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release
```

Next, include an alias that includes the path to the directory where the shared library is located:
```clojure
;; in aliases
;; add jvm opt for local llama build.
:local-llama {:jvm-opts ["-Djna.library.path=/path/to/llama.cpp/build/"]}
```

##### Dual wielding with local builds

It's possible to use both ggml and gguf models in the same process (ie. "dual wielding"). The trick is to treat the older ggml llama.cpp version and the newer gguf llama.cpp versions as separate libraries. Each shared library must have a unique name. If using only one of the ggml or gguf formats is required, then using the libllama.(so,dylib) is sufficient. For dual wielding, the ggml version should be called libllama.(so,dylib) and the gguf version should be renamed to libllama-gguf.(so,dylib). Further, the soname of the gguf version must also be updated. For example:

Linux
```bash
mv libllama.so libllama-gguf.so
sudo apt-get install patchelf
patchelf --set-soname libllama-gguf.so libllama-gguf.so
```

Mac OSX
```bash
mv libllama.dylib libllama-gguf.dylib
install_name_tool -id libllama-gguf.dylib libllama-gguf.dylib
```

### Obtaining models

For more complete information about the models that llama.clj can work with, refer to the [llama.cpp readme](https://github.com/ggerganov/llama.cpp).

Another good resource for models is [TheBloke](https://huggingface.co/TheBloke) on [huggingface](https://huggingface.co/).

## Cli Usage

```sh
clojure -M -m com.phronemophobic.llama <path-to-model> <prompt>
```
Example:

```bash
mkdir -p models
# Download 3.6Gb model to models/ directory
(cd models && curl -L -O 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin')
clojure -M:mvn-llama -m com.phronemophobic.llama "models/llama-2-7b-chat.ggmlv3.q4_0.bin" "what is 2+2?"
```

## cuBLAS support

For gpu support on linux, cuda must be installed. The instructions for cuda installation can be found in [nvidia's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html).

Make sure to restart and follow the [post installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) so that the cuda development tools like nvcc are available on the path.

Currently, pre-compiled binaries of llama.cpp with cuBLAS support are not available. The llama.cpp native dependencies must be [compiled locally](#locally-compiled) with `-DLLAMA_CUBLAS=ON` as argument. Something like:

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 4329d1acb01c353803a54733b8eef9d93d0b84b2
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON ..
cmake --build . --config Release
```

### More cuBLAS Resources
- https://github.com/ggerganov/llama.cpp#cublas
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html

## "Roadmap"

- [ ] Add convenience namespace for obtaining and using models.
- [ ] Pure clojure implementation for mirostatv2 and other useful samplers.
- [ ] Provide reasonable default implementations for generating responses larger than the context size.
- [X] Update llama.cpp to [support gguf format](https://github.com/phronmophobic/llama.clj/issues/8)
- [X] More docs!
  - [X] Reference docs
  - [X] Intro Guide to LLMs.

## License

The MIT License (MIT)

Copyright © 2023 Adrian Smith

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



