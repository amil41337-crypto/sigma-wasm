# Sigma WASM - WebAssembly Demos

A collection of interactive WebAssembly demos showcasing Rust-compiled WASM modules integrated with modern web technologies. This project demonstrates various use cases including pathfinding algorithms, computer vision, natural language processing, and generative art.

Check out the live demos <a href="https://wasm-demos.onrender.com" target="_blank" rel="noopener noreferrer">here</a>.

## Demo Endpoints

### A* Pathfinding (`/astar`)

An interactive pathfinding algorithm visualization built with Rust and WebAssembly. This demo provides real-time visualization of the A* pathfinding algorithm with interactive controls.

**Technology Stack:**
- Rust/WASM for pathfinding logic
- Canvas-based rendering
- Real-time game loop with `requestAnimationFrame`

**Features:**
- Real-time pathfinding visualization on procedurally generated maps
- Interactive controls: arrow keys or mouse to move the starting point
- Spacebar to randomize the map
- FPS counter and path count display
- Multi-layer canvas rendering system

**WASM Module:** `wasm-astar`

**Key Files:**
- Route: [`src/routes/astar.ts`](src/routes/astar.ts)
- Rust Source: [`wasm-astar/src/lib.rs`](wasm-astar/src/lib.rs)
- HTML: [`pages/astar.html`](pages/astar.html)

This demo is a port of an A* implementation by [Jacob Reichert](https://github.com/jacobdeichert/wasm-astar), demonstrating how to manage game state in Rust using global mutexes and handle the complexities of WASM-JS interop.

---

### SmolVLM-500M (`/preprocess-smolvlm-500m`)

Visual Question Answering (VQA) and Image Captioning using the SmolVLM-500M-Instruct model. This demo showcases client-side vision-language AI with high-performance image preprocessing.

**Technology Stack:**
- Rust/WASM for image preprocessing (Lanczos3 resizing)
- ONNX Runtime Web for model inference
- Hugging Face model hosting

**Features:**
- Image preprocessing with WASM (resizing to 224×224 for model input)
- Visual Question Answering: answer questions about image content
- Image Captioning: generate descriptive text from images
- Real-time filter preview with sliders (contrast, cinematic effects)
- Webcam support for live image capture
- Model caching for faster subsequent loads

**WASM Module:** `wasm-preprocess`

**Model:** SmolVLM-500M-Instruct from Hugging Face

**Key Files:**
- Route: [`src/routes/preprocess-smolvlm-500m.ts`](src/routes/preprocess-smolvlm-500m.ts)
- Model Integration: [`src/models/smolvlm.ts`](src/models/smolvlm.ts)
- HTML: [`pages/preprocess-smolvlm-500m.html`](pages/preprocess-smolvlm-500m.html)

The WASM preprocessing module handles efficient image decoding and resizing, preparing images for the vision encoder. The model runs entirely client-side using ONNX Runtime Web with WASM acceleration.

---

### SmolVLM-256M (`/preprocess-smolvlm-256m`)

A faster, smaller variant of the SmolVLM demo using the 256M parameter model. Optimized for speed while maintaining similar capabilities to the 500M version.

**Technology Stack:**
- Rust/WASM for image preprocessing
- ONNX Runtime Web for model inference
- SmolVLM-256M model (512×512 input size)

**Features:**
- Same VQA and captioning capabilities as 500M version
- Faster inference due to smaller model size
- Optimized preprocessing pipeline
- Real-time filters and webcam support
- Lower memory footprint

**WASM Module:** `wasm-preprocess-256m`

**Model:** SmolVLM-256M (uses 512×512 input resolution)

**Key Files:**
- Route: [`src/routes/preprocess-smolvlm-256m.ts`](src/routes/preprocess-smolvlm-256m.ts)
- Model Integration: [`src/models/smolvlm-256m.ts`](src/models/smolvlm-256m.ts)
- HTML: [`pages/preprocess-smolvlm-256m.html`](pages/preprocess-smolvlm-256m.html)

This demo is ideal for devices with limited memory or when faster response times are preferred. The preprocessing pipeline is specifically optimized for the 512×512 input size.

---

### ViT-GPT2 Image Captioning (`/image-captioning`)

Image captioning using a Vision Transformer (ViT) encoder with GPT-2 decoder, powered by Transformers.js. This demo showcases a different approach to vision-language models compared to SmolVLM.

**Technology Stack:**
- Rust/WASM for image preprocessing and filters
- Transformers.js for model inference
- ViT-GPT2 model architecture

**Features:**
- Image captioning: generate natural language descriptions
- WASM preprocessing with multiple filter options (contrast, cinematic, sepia)
- Real-time filter preview with sliders
- Webcam support for live capture
- Client-side inference with no server calls

**WASM Module:** `wasm-preprocess-image-captioning`

**Model:** ViT-GPT2 via Transformers.js

**Key Files:**
- Route: [`src/routes/image-captioning.ts`](src/routes/image-captioning.ts)
- Model Integration: [`src/models/image-captioning.ts`](src/models/image-captioning.ts)
- HTML: [`pages/image-captioning.html`](pages/image-captioning.html)

The ViT-GPT2 model combines a Vision Transformer for image understanding with GPT-2 for text generation, providing a different architectural approach compared to the SmolVLM models. All processing happens client-side using WebAssembly acceleration.

---

### Function Calling Agent (`/function-calling`)

A client-side autonomous agent with local LLM inference and function calling capabilities. This demo showcases how to build goal-oriented agents that can use tools to accomplish tasks.

**Technology Stack:**
- Transformers.js for LLM inference
- Rust/WASM for tool execution
- DistilGPT-2 model for text generation

**Features:**
- Goal-oriented agent execution: describe a goal and the agent plans steps
- Function calling: agent can call WASM tools (calculate, process_text, get_stats)
- Human-in-the-loop clarification: agent asks for clarification when needed
- Step-by-step execution display showing reasoning process
- Tool execution results feed back into agent reasoning

**WASM Module:** `wasm-agent-tools`

**Model:** DistilGPT-2 via Transformers.js

**Available Tools:**
- `calculate(expression)`: Evaluate mathematical expressions
- `process_text(text, operation)`: Text processing (uppercase, lowercase, reverse, length, word_count)
- `get_stats(data)`: Statistical analysis of data arrays

**Key Files:**
- Route: [`src/routes/function-calling.ts`](src/routes/function-calling.ts)
- Model Integration: [`src/models/function-calling.ts`](src/models/function-calling.ts)
- HTML: [`pages/function-calling.html`](pages/function-calling.html)

The agent analyzes the user's goal, decides which tools to use, executes them via WASM, and uses the results to generate a final response. All processing happens client-side, demonstrating fully autonomous agents running in the browser.

---

### Fractal Chat (`/fractal-chat`)

An interactive chat interface that generates fractal images based on keywords in your messages. When you mention a fractal type, a corresponding image is generated and displayed. Otherwise, the chat model responds conversationally.

**Technology Stack:**
- Rust/WASM for fractal generation
- Transformers.js for chat model inference
- Qwen1.5-0.5B-Chat model

**Features:**
- Keyword detection for fractal generation
- 8 fractal types: Mandelbrot, Julia, Buddhabrot, Orbit-Trap, Gray-Scott, L-System, Flames, Strange Attractors
- Conversational AI responses when no fractal keyword is detected
- Real-time fractal generation (512×512 images)
- Chat history with images embedded in conversation

**WASM Module:** `wasm-fractal-chat`

**Model:** Qwen1.5-0.5B-Chat via Transformers.js

**Fractal Keywords:**
- `fractal` - Random fractal from all types
- `mandelbrot` - Classic Mandelbrot Set
- `julia` - Julia Sets
- `buddhabrot` or `nebulabrot` - Buddhabrot/Nebulabrot
- `orbit-trap` - Orbit-Trap Fractals
- `gray-scott`, `reaction`, or `diffusion` - Gray-Scott Reaction-Diffusion
- `l-system`, `tree`, or `plant` - L-System Fractals
- `flames` - Fractal Flames
- `strange`, `attractors`, `lorenz`, `clifford`, or `de jong` - Strange Attractors

**Key Files:**
- Route: [`src/routes/fractal-chat.ts`](src/routes/fractal-chat.ts)
- Rust Source: [`wasm-fractal-chat/src/lib.rs`](wasm-fractal-chat/src/lib.rs)
- HTML: [`pages/fractal-chat.html`](pages/fractal-chat.html)

The demo combines generative art with conversational AI, creating a unique interactive experience. Fractals are generated in real-time using optimized Rust algorithms compiled to WASM, while the chat model provides natural language interaction.

---

### Hello WASM - Student Template (`/hello-wasm`)

A simplified template designed for students to learn from. This demo demonstrates the core WASM state management pattern used throughout the project, making it an ideal starting point for understanding Rust-WASM integration.

**Technology Stack:**
- Rust/WASM for state management
- TypeScript for client-side integration
- Simple HTML/CSS for UI

**Features:**
- Counter increment/decrement functionality
- Message display and editing
- Demonstrates `LazyLock<Mutex<State>>` pattern in Rust
- Shows how to export functions with `#[wasm_bindgen]`
- TypeScript integration with proper type safety
- Route handler pattern demonstration

**WASM Module:** `wasm-hello`

**Key Learning Points:**
- State management using `LazyLock<Mutex<State>>` in Rust
- WASM bindings with `#[wasm_bindgen]`
- TypeScript type definitions and validation
- Route handler initialization pattern
- UI event handling with WASM functions

**Key Files:**
- Route: [`src/routes/hello-wasm.ts`](src/routes/hello-wasm.ts)
- Rust Source: [`wasm-hello/src/lib.rs`](wasm-hello/src/lib.rs)
- HTML: [`pages/hello-wasm.html`](pages/hello-wasm.html)

This template is intentionally simplified to serve as a learning resource. Students can extend it by adding new state fields, WASM functions, and UI elements following the established patterns.

---

### Babylon WFC - Wave Function Collapse 3D (`/babylon-wfc`)

A 3D visualization of the Wave Function Collapse (WFC) algorithm, a procedural generation technique that creates coherent patterns from a set of rules. This demo showcases 5 simple tile types rendered in 3D using BabylonJS with mesh instancing for optimal performance. The layout uses a hexagonal grid pattern with hash map storage for efficient sparse grid management and no size limitations.

**Technology Stack:**
- Rust/WASM for WFC algorithm implementation
- BabylonJS for 3D rendering
- Mesh instancing for performance optimization
- Babylon 2D UI for controls
- Qwen 1.5-0.5B-Chat (Transformers.js) for text-to-layout generation

**Features:**
- Wave Function Collapse algorithm generating hexagonal tile grids
- Layer-based hexagon: 30 layers (2791 tiles total) forming a complete hexagon shape
- Hexagon pattern: Each layer adds a ring around the center (layer 0 = 1 tile, layer n adds 6n tiles)
- Total tiles formula: 3n(n+1) + 1, where n = 30 → 2791 tiles
- 5 simple tile types: Grass, Building, Road, Forest, and Water
- 3D visualization with interactive camera controls
- Mesh instancing for efficient rendering of thousands of hexagonal tiles
- Babylon 2D UI buttons for recompute and fullscreen
- **Text-to-Layout Generation**: Enter natural language prompts (e.g., "sparse buildings", "dense clustered layout") to guide WFC generation
- Real-time procedural generation

**WASM Module:** `wasm-babylon-wfc`

**Tile Types:**
1. Grass - Natural terrain areas
2. Building - Structures and constructions
3. Road - Pathways and routes
4. Forest - Trees and vegetation
5. Water - Water bodies and lakes

**WFC Algorithm Deep Dive:**

The Wave Function Collapse algorithm is a constraint-based procedural generation technique inspired by quantum mechanics. Each cell starts in a "superposition" of all possible tile types, then collapses to a single state based on constraints from neighboring cells.

**Key Concepts:**

1. **Superposition**: Each cell maintains a list of possible tile types (the "wave function"). Initially, all 5 tile types are possible for each cell.

2. **Entropy**: The number of possible tile types for a cell. Lower entropy means fewer possibilities, making the cell more "certain" about its final state. The algorithm always collapses the cell with the lowest entropy first to minimize contradictions.

3. **Constraint Propagation**: When a cell is collapsed to a specific tile type, adjacency rules define what neighboring cells can be. All incompatible tile types are removed from the neighbor's wave function, and this propagation continues recursively to all affected neighbors.

4. **Adjacency Rules**: The system is prepared for adjacency constraints where certain tile types can only be placed next to specific other types. For example:
   - Roads should connect to buildings or other roads
   - Water should be adjacent to grass or other water
   - Buildings can be adjacent to roads or grass
   - Forests can be adjacent to grass or other forests

5. **Pre-Constraints**: Before WFC begins, certain cells can be "pre-collapsed" to specific tile types. This is used for:
   - Direct tile type assignment based on layout constraints
   - Text-to-layout generation (user-specified constraints)
   - Guided generation with specific terrain patterns

**Two-Phase Generation:**

- **Phase 1 - Hexagon Pattern Generation**: 
  - Grid is a complete hexagon shape with 30 layers (2791 tiles)
  - Hexagon layers: Layer 0 = 1 tile (center), layer n adds 6n tiles
  - Total tiles: 3×30×31 + 1 = 2791 tiles
  - Only tiles within the hexagon pattern are generated
  - Tile types are assigned based on layout constraints (grass ratio, building density, etc.)
  - Pre-constraints are set before WFC begins using hash map storage
  - This creates a natural-looking hexagon-shaped map

- **Phase 2 - WFC Collapse**: 
  - All cells start in superposition (all 5 tile types possible)
  - Pre-constrained cells are already collapsed, so their constraints propagate immediately
  - Algorithm loop:
    1. Find the uncollapsed cell with lowest entropy
    2. If multiple cells have the same lowest entropy, pick randomly
    3. Collapse the cell to a random valid tile type from its possibilities
    4. Propagate constraints to all neighbors (remove incompatible tiles)
    5. Repeat until all cells are collapsed
  - If a cell has 0 valid possibilities (contradiction), it falls back to `Grass`
  - After the loop, any remaining uncollapsed cells are filled with `Grass` to prevent gaps

**Text-to-Layout Workflow (TileGPT-Inspired):**

This feature combines natural language understanding with constraint-based generation, inspired by the [TileGPT paper](https://tilegpt.github.io/).

1. **User Input**: User enters a text prompt (e.g., "sparse buildings", "dense clustered layout", "many small buildings")

2. **Qwen Chat Model Generation**:
   - Uses `Xenova/qwen1.5-0.5b-chat` (a chat-optimized model, better at instruction following than base models)
   - Chat template format enables structured output
   - Generates a JSON description with:
     - `buildingDensity`: "sparse" | "medium" | "dense" (controls number of buildings)
     - `clustering`: "clustered" | "distributed" | "random" (controls spatial distribution)
     - `grassRatio`: number between 0.0-1.0 (percentage of grid that should be grass)
     - `buildingSizeHint`: "small" | "medium" | "large" (hint for building dimensions)

3. **Constraint Parsing**:
   - First attempts JSON parsing (chat models are better at structured output)
   - Falls back to regex pattern matching if JSON parsing fails
   - Defaults to reasonable values if parsing completely fails

4. **Pre-Constraint Conversion**:
   - **Tile Type Assignment**: Directly assigns tile types to hexagon cells based on constraints
   - **Distribution**: Randomly distributes tile types (grass, building, road, forest, water) based on `grassRatio` and other constraints
   - **Hash Map Storage**: All pre-constraints are stored in a hash map with hex coordinates (q, r) as keys, enabling O(1) lookups and no size limitations

5. **WFC Generation**: WFC algorithm runs with pre-constraints applied, generating a detailed layout that respects the high-level constraints

6. **3D Visualization**: BabylonJS renders the result with color-coded tiles and interactive camera controls

**Why Qwen Chat Model?**

Unlike base language models (like DistilGPT-2), Qwen is a chat model specifically fine-tuned for:
- **Better instruction following**: Understands and follows structured prompts more reliably
- **Structured output**: More likely to generate valid JSON when requested
- **Contextual understanding**: Better at interpreting nuanced layout descriptions
- **Chat template format**: Uses proper message formatting for more reliable responses

This makes it ideal for converting natural language into structured constraint data.

**Visual Features:**
- **Color Coding**: Grass (green), Building (gray), Road (dark gray), Forest (dark green), Water (blue)
- **Camera**: Initially centered on hexagon grid with optimal viewing angle
- **Interactive**: Mouse controls for rotation and zoom

**Key Files:**
- Route: [`src/routes/babylon-wfc.ts`](src/routes/babylon-wfc.ts)
- Rust Source: [`wasm-babylon-wfc/src/lib.rs`](wasm-babylon-wfc/src/lib.rs)
- HTML: [`pages/babylon-wfc.html`](pages/babylon-wfc.html)

The WFC algorithm ensures that tiles are placed according to edge compatibility rules, creating visually coherent patterns. The 3D visualization uses BabylonJS mesh instancing to efficiently render thousands of tiles while maintaining smooth performance.

**Architecture Patterns:**

This project demonstrates several important patterns for Rust WASM integration:

1. **State Management Pattern** (`LazyLock<Mutex<State>>`):
   ```rust
   static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::new()));
   ```
   - **Why?**: WASM modules are stateless by default, but many algorithms need mutable state
   - **Learning Point**: `LazyLock` provides thread-safe lazy initialization, `Mutex` ensures safe concurrent access
   - **Trade-off**: Mutex adds overhead but enables safe shared state in WASM

2. **WASM-JS Interop Pattern**:
   - TypeScript interfaces define expected WASM exports
   - Runtime validation ensures type safety without type assertions
   - Dynamic imports with validation before use
   - **Learning Point**: Never trust dynamic imports - always validate at runtime

3. **Route Handler Pattern**:
   - Each endpoint has: route handler (TS) + HTML page + WASM module
   - Lazy loading: WASM modules only load when route is accessed
   - Error handling: Graceful degradation with user-friendly messages
   - **Learning Point**: Separation of concerns makes code maintainable and testable

4. **Type Safety Patterns**:
   - Discriminated unions for tile types and edge types
   - Type guards instead of type assertions
   - Runtime validation before type narrowing
   - **Learning Point**: TypeScript's type system is powerful but requires discipline to use effectively

5. **Hash Map Storage Pattern**:
   ```rust
   struct WfcState {
       grid: HashMap<(i32, i32), TileType>,
       pre_constraints: HashMap<(i32, i32), TileType>,
   }
   ```
   - **Why?**: Enables sparse grid storage with O(1) lookups and no size limitations
   - **Learning Point**: Hash maps are ideal for hexagonal grids where not all coordinates are used
   - **Trade-off**: Slightly more memory overhead than arrays, but enables unlimited grid sizes
   - **Benefits**: No bounds checking needed, memory efficient for sparse grids, scales to any hexagon size

---

## Technical Architecture

### Technology Stack

**Core Technologies:**

- **Rust**: Systems programming language compiled to WebAssembly
  - **Why Rust?**: Memory safety, performance, and excellent WASM tooling
  - **Learning Point**: Rust's ownership system prevents memory bugs while compiling to efficient WASM
  - **When to Use WASM**: For computationally intensive tasks (image processing, algorithms, game logic)
  - **Trade-off**: WASM adds build complexity but provides near-native performance

- **TypeScript**: Type-safe frontend development
  - **Why TypeScript?**: Catches errors at compile-time, improves developer experience
  - **Learning Point**: Strict typing prevents runtime errors and makes code more maintainable
  - **Integration Pattern**: TypeScript interfaces define contracts with WASM modules

- **Vite**: Fast build tool and dev server
  - **Why Vite?**: Fast HMR (Hot Module Replacement), optimized production builds
  - **Learning Point**: Vite's plugin system allows custom WASM handling and routing

- **wasm-bindgen**: Rust-WASM interop
  - **Why wasm-bindgen?**: Automatically generates JavaScript bindings from Rust code
  - **Learning Point**: `#[wasm_bindgen]` macro exports Rust functions to JavaScript with type safety

**AI/ML Frameworks:**

- **ONNX Runtime Web**: For SmolVLM models (WASM/WebGPU acceleration)
  - **Why ONNX?**: Industry-standard format, optimized inference engines
  - **Learning Point**: ONNX models can run in browsers with hardware acceleration
  - **Trade-off**: Larger model files but better performance than pure JavaScript

- **Transformers.js**: For ViT-GPT2, DistilGPT-2, and Qwen models
  - **Why Transformers.js?**: Easy-to-use API, automatic quantization, browser-optimized
  - **Learning Point**: Chat models (like Qwen) are better at instruction following than base models
  - **Trade-off**: Smaller models run faster but may have lower quality than larger models

- **@huggingface/tokenizers**: Tokenization for language models
  - **Why Separate Tokenizer?**: Consistent tokenization across different model backends
  - **Learning Point**: Tokenization converts text to model input format

**3D Rendering:**

- **BabylonJS**: 3D rendering engine
  - **Why BabylonJS?**: Full-featured, well-documented, active community
  - **Learning Point**: Mesh instancing allows rendering thousands of objects efficiently
  - **Performance**: Instancing reduces draw calls from 2791 to 5 (one per tile type)

**Build & Deployment:**

- **Docker**: Containerized builds and deployment
  - **Why Docker?**: Reproducible builds, consistent environments
  - **Learning Point**: Multi-stage builds optimize image size and build caching

- **nginx**: Static file serving in production
  - **Why nginx?**: Fast, reliable, good caching support
  - **Learning Point**: Proper MIME types and caching headers are critical for WASM

- **Render.com**: Hosting platform
  - **Why Render?**: Easy Docker deployment, automatic SSL, health checks
  - **Learning Point**: `render.yaml` enables infrastructure-as-code deployment

### WASM Modules Organization

The project uses a Rust workspace with multiple WASM crates, each compiled independently:

- `wasm-astar`: Pathfinding algorithm and game state management
- `wasm-preprocess`: Image preprocessing for SmolVLM-500M (224×224)
- `wasm-preprocess-256m`: Image preprocessing for SmolVLM-256M (512×512)
- `wasm-preprocess-image-captioning`: Image preprocessing and filters for ViT-GPT2
- `wasm-agent-tools`: Tool functions for the agent (calculate, process_text, get_stats)
- `wasm-fractal-chat`: Fractal generation algorithms
- `wasm-hello`: Student template demonstrating WASM state management
- `wasm-babylon-wfc`: Wave Function Collapse algorithm for procedural generation

Each module is built using `wasm-bindgen` and optimized with `wasm-opt` for smaller binary sizes.

### Routing System

The application uses a client-side router (`src/main.ts`) that:
- Detects the current pathname
- Lazy-loads the appropriate route handler
- Initializes the corresponding WASM module and UI
- Handles errors gracefully with user-friendly messages

Routes are defined in `src/main.ts` and each route has:
- A TypeScript route handler (`src/routes/*.ts`)
- An HTML page (`pages/*.html`)
- A corresponding WASM module

### Model Loading Strategies

**ONNX Runtime Models (SmolVLM):**
- Models are downloaded from Hugging Face
- Cached using the Cache API for faster subsequent loads
- CORS proxies are used when direct access fails
- Progress tracking during download and initialization

**Transformers.js Models:**
- Models are loaded on-demand via Transformers.js
- Automatic quantization and optimization
- WebAssembly acceleration for inference
- Model files are cached by the browser

**Error Handling:**
- Graceful degradation if models fail to load
- Detailed error messages for debugging
- Fallback to preprocessing-only mode when models unavailable

---

## Building

### Local Development (Without Docker)

#### Quick Setup

Run the setup script to install all required dependencies:

```bash
./scripts/setup-local.sh
```

This will:
- Check for Rust, Node.js, and npm
- Install wasm-bindgen-cli if missing
- Install npm dependencies
- Set up the wasm32-unknown-unknown target

#### Manual Setup

If you prefer to set up manually:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add wasm32 target
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen-cli
cargo install wasm-bindgen-cli --version 0.2.87

# Install wasm-opt (optional but recommended)
# On macOS: brew install binaryen
# On Debian/Ubuntu: sudo apt-get install binaryen
# On Alpine: apk add binaryen
# Or via npm: npm install -g wasm-opt

# Install npm dependencies
npm install
```

#### Development

Start the development server:

```bash
# Option 1: Use the dev script
./scripts/dev-local.sh

# Option 2: Use npm directly
npm run dev
```

#### Production Build

Build for production:

```bash
# Build WASM and frontend
npm run build

# Or build WASM only
npm run build:wasm

# Preview production build
npm run preview
```

### Docker Build

#### Build Docker Image

```bash
# Build the Docker image
npm run build:docker
# Or directly:
docker build -t sigma-wasm .
```

#### Run Docker Container

```bash
# Run the container
docker run -p 3000:80 sigma-wasm

# Access at http://localhost:3000
```

#### Docker Compose (Optional)

If you have `docker-compose.yml`:

```bash
docker-compose up
```

## Deployment

### Render.com Deployment

This project is configured for automatic deployment on Render.com using Docker.

#### Prerequisites

1. A Render.com account
2. A Git repository (GitHub, GitLab, or Bitbucket)
3. The repository connected to Render.com

#### Automatic Deployment

1. **Push your code to Git** - Ensure `render.yaml` is in the root directory
2. **Connect to Render** - In Render dashboard, create a new "Blueprint" service
3. **Render will automatically:**
   - Detect the `render.yaml` file
   - Build using the Dockerfile
   - Deploy the service
   - Set up auto-deploy from your Git repository

#### Manual Configuration

If you prefer to configure manually:

1. Create a new **Web Service** in Render
2. Connect your Git repository
3. Set the following:
   - **Environment**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Context**: `.`
   - **Build Command**: (auto-detected from Dockerfile)
   - **Start Command**: (auto-detected from Dockerfile)

#### Environment Variables

Environment variables can be set in:
- `render.yaml` (for static values)
- Render.com dashboard (for secrets and dynamic values)

See `.env.example` for available environment variables.

#### Build Configuration

The `render.yaml` file includes:
- Build filters (only rebuild on relevant file changes)
- Health check configuration
- Auto-deploy settings
- Environment variables

### Other Deployment Options

#### Static File Hosting

After building with `npm run build`, the `dist/` directory contains static files that can be served by:
- Any static file server (nginx, Apache, etc.)
- CDN services (Cloudflare, AWS CloudFront, etc.)
- Static hosting (Netlify, Vercel, GitHub Pages, etc.)

```bash
# Build the project
npm run build

# The dist/ directory contains all static files
# Serve with any static file server:
npx serve dist
```

## Environment Variables

See `.env.example` for a template of available environment variables.

### Build-time Variables

- `NODE_ENV` - Set to `production` for production builds

### Runtime Variables

Currently, no runtime environment variables are required. Add them to `.env.example` and `render.yaml` as needed.

## Troubleshooting

### Build Issues

**Error: `cargo: command not found`**
- Install Rust: https://rustup.rs/
- Ensure Rust is in your PATH

**Error: `wasm-bindgen: command not found`**
- Install with: `cargo install wasm-bindgen-cli --version 0.2.87`
- Ensure `~/.cargo/bin` is in your PATH

**Error: `wasm-opt: command not found`**
- This is optional but recommended
- Install via package manager or npm (see setup instructions above)
- Build will still work without it, but WASM won't be optimized

**Docker build fails**
- Ensure Docker is running
- Check that all required files are present
- Review Docker build logs for specific errors

### Runtime Issues

**WASM module not loading**
- Check browser console for errors
- Ensure `pkg/` directory is accessible
- Verify wasm-bindgen output files are present

**404 errors for assets**
- Ensure Vite build completed successfully
- Check that `dist/` directory contains all files
- Verify nginx configuration (if using Docker)

**Model loading fails**
- Check browser console for CORS errors
- Verify internet connection (models download from Hugging Face)
- Try clearing browser cache
- Check that CORS proxies are accessible

### Render.com Issues

**Deployment fails**
- Check Render build logs
- Verify `render.yaml` syntax
- Ensure Dockerfile is valid
- Check that all required files are in the repository

**Service not starting**
- Check Render service logs
- Verify health check endpoint
- Ensure port 80 is exposed in Dockerfile

## Project Structure

```
sigma-wasm/
├── Dockerfile              # Multi-stage Docker build
├── .dockerignore           # Docker build exclusions
├── render.yaml             # Render.com configuration
├── .env.example            # Environment variables template
├── Cargo.toml              # Rust workspace configuration
├── package.json            # Node.js dependencies
├── vite.config.ts          # Vite configuration
├── tsconfig.json           # TypeScript configuration
├── scripts/
│   ├── build.sh            # WASM build script
│   ├── build-wasm.sh       # WASM build script
│   ├── setup-local.sh      # Local setup script
│   └── dev-local.sh        # Local dev server script
├── src/
│   ├── main.ts             # TypeScript entry point and router
│   ├── types.ts            # TypeScript type definitions
│   ├── styles.css          # Global styles
│   ├── routes/             # Route handlers for each demo
│   │   ├── astar.ts
│   │   ├── preprocess-smolvlm-500m.ts
│   │   ├── preprocess-smolvlm-256m.ts
│   │   ├── image-captioning.ts
│   │   ├── function-calling.ts
│   │   ├── fractal-chat.ts
│   │   ├── hello-wasm.ts
│   │   └── babylon-wfc.ts
│   ├── models/             # Model integration code
│   │   ├── smolvlm.ts
│   │   ├── smolvlm-256m.ts
│   │   ├── image-captioning.ts
│   │   └── function-calling.ts
│   ├── wasm/               # WASM loader utilities
│   │   ├── loader.ts
│   │   └── types.ts
│   └── [rust modules]      # Shared Rust source (if any)
├── pages/                  # HTML pages for each demo
│   ├── astar.html
│   ├── preprocess-smolvlm-500m.html
│   ├── preprocess-smolvlm-256m.html
│   ├── image-captioning.html
│   ├── function-calling.html
│   ├── fractal-chat.html
│   ├── hello-wasm.html
│   └── babylon-wfc.html
├── wasm-astar/             # A* pathfinding WASM crate
│   ├── Cargo.toml
│   └── src/lib.rs
├── wasm-preprocess/        # Image preprocessing WASM crate (500M)
│   ├── Cargo.toml
│   └── src/lib.rs
├── wasm-preprocess-256m/   # Image preprocessing WASM crate (256M)
│   ├── Cargo.toml
│   └── src/lib.rs
├── wasm-preprocess-image-captioning/  # Image preprocessing WASM crate (ViT-GPT2)
│   ├── Cargo.toml
│   └── src/lib.rs
├── wasm-agent-tools/       # Agent tools WASM crate
│   ├── Cargo.toml
│   └── src/lib.rs
├── wasm-fractal-chat/      # Fractal generation WASM crate
│   ├── Cargo.toml
│   └── src/lib.rs
├── wasm-hello/             # Student template WASM crate
│   ├── Cargo.toml
│   └── src/lib.rs
├── wasm-babylon-wfc/       # Wave Function Collapse WASM crate
│   ├── Cargo.toml
│   └── src/lib.rs
├── pkg/                    # Compiled WASM modules (generated)
│   ├── wasm_astar/
│   ├── wasm_preprocess/
│   ├── wasm_preprocess_256m/
│   ├── wasm_preprocess_image_captioning/
│   ├── wasm_agent_tools/
│   ├── wasm_fractal_chat/
│   ├── wasm_hello/
│   └── wasm_babylon_wfc/
└── dist/                   # Production build output (gitignored)
```

## Learning Resources

### Wave Function Collapse Algorithm

- **Original Paper**: [WaveFunctionCollapse by Maxim Gumin](https://github.com/mxgmn/WaveFunctionCollapse)
- **Algorithm Explanation**: [WFC Algorithm Overview](https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/)
- **Academic Paper**: [WaveFunctionCollapse: Constraint Solving in the Wild](https://adamsmith.as/papers/wfc_is_constraint_solving_in_the_wild.pdf)
- **TileGPT Paper**: [Generative Design through Quality-Diversity Data Synthesis and Language Models](https://tilegpt.github.io/)

### WebAssembly (WASM)

- **Rust WASM Book**: [The Rust and WebAssembly Book](https://rustwasm.github.io/docs/book/)
- **wasm-bindgen Guide**: [wasm-bindgen Documentation](https://rustwasm.github.io/wasm-bindgen/)
- **MDN WebAssembly Guide**: [WebAssembly Concepts](https://developer.mozilla.org/en-US/docs/WebAssembly/Concepts)
- **Performance Guide**: [WebAssembly Performance](https://web.dev/webassembly/)

### BabylonJS

- **Official Documentation**: [Babylon.js Documentation](https://doc.babylonjs.com/)
- **Getting Started**: [Babylon.js Getting Started Guide](https://doc.babylonjs.com/setup/support/gettingStarted)
- **Mesh Instancing**: [Instanced Meshes Tutorial](https://doc.babylonjs.com/features/featuresDeepDive/mesh/copies/instances)
- **2D UI**: [Babylon.js GUI Documentation](https://doc.babylonjs.com/features/featuresDeepDive/gui/gui)

### Transformers.js

- **Official Documentation**: [Transformers.js Documentation](https://huggingface.co/docs/transformers.js/)
- **GitHub Repository**: [Transformers.js on GitHub](https://github.com/xenova/transformers.js)
- **Model Hub**: [Hugging Face Model Hub](https://huggingface.co/models)

### TypeScript

- **TypeScript Handbook**: [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- **Discriminated Unions**: [TypeScript Discriminated Unions](https://www.typescriptlang.org/docs/handbook/2/narrowing.html#discriminated-unions)
- **Type Guards**: [Type Guards and Differentiating Types](https://www.typescriptlang.org/docs/handbook/2/narrowing.html#using-type-predicates)

### Rust

- **The Rust Book**: [The Rust Programming Language Book](https://doc.rust-lang.org/book/)
- **Rust by Example**: [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- **Rust WASM Working Group**: [Rust WASM WG](https://rustwasm.github.io/)

### Build Tools

- **Vite Documentation**: [Vite Guide](https://vitejs.dev/guide/)
- **Docker Documentation**: [Docker Documentation](https://docs.docker.com/)
- **Nginx Documentation**: [Nginx Documentation](https://nginx.org/en/docs/)
