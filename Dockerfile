# Multi-stage Dockerfile for Rust WASM + Vite build
# Stage 1: Rust WASM Builder
# Using rust:alpine (smallest image, ~500MB) instead of rust:1-alpine
FROM rust:alpine AS rust-builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    perl \
    make \
    git \
    bash

# Install wasm-bindgen-cli
# Version must match wasm-bindgen crate version in Cargo.toml
# Using 0.2.106 which is compatible with Rust 1.71+ and matches "0.2" in Cargo.toml
RUN cargo install wasm-bindgen-cli --version 0.2.106

# Install wasm-opt from binaryen
RUN apk add --no-cache binaryen

# Set working directory
WORKDIR /app

# Copy Cargo files for dependency caching
COPY Cargo.toml ./

# Create dummy src to cache dependencies (if Cargo.lock exists)
RUN mkdir -p src && \
    echo "fn main() {}" > src/lib.rs || true

# Build dependencies only (for caching)
RUN cargo build --target wasm32-unknown-unknown --release || true

# Copy actual source code
COPY src ./src
COPY scripts ./scripts

# Make build script executable
RUN chmod +x scripts/build.sh

# Add wasm32 target
RUN rustup target add wasm32-unknown-unknown

# Build WASM module
RUN ./scripts/build.sh

# Stage 2: Node.js Frontend Builder
FROM node:20-alpine AS node-builder

WORKDIR /app

# Copy package files for dependency caching
COPY package.json ./

# Install dependencies
# Using npm install instead of npm ci since package-lock.json is gitignored
# This will generate a new package-lock.json during build
RUN npm install

# Copy Rust build output
COPY --from=rust-builder /app/pkg ./pkg

# Copy frontend source
COPY src ./src
COPY index.html ./
COPY vite.config.ts ./
COPY tsconfig.json ./

# Build frontend
# WASM is already built in stage 1, so just run vite build (skip build:wasm)
RUN npx vite build

# Stage 3: Runtime (nginx for static files)
FROM nginx:alpine AS runtime

# Install gettext for envsubst (dynamic port substitution) and wget for health checks
RUN apk add --no-cache gettext wget

# Copy built static files
COPY --from=node-builder /app/dist /usr/share/nginx/html

# Copy main nginx configuration with performance optimizations
COPY nginx-main.conf /etc/nginx/nginx.conf

# Copy nginx server configuration template (will be processed with envsubst)
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Copy custom entrypoint script to handle PORT default
COPY docker-entrypoint.sh /docker-entrypoint-custom.sh
RUN chmod +x /docker-entrypoint-custom.sh

# nginx:alpine processes templates in /etc/nginx/templates/ automatically
# Our custom entrypoint ensures PORT has a default value for local testing
# Render.com will always provide PORT environment variable

# Expose port (Render.com will set PORT env var)
EXPOSE 80

# Health check endpoint at /health (Render.com uses healthCheckPath: /)
# Using /health endpoint for Docker HEALTHCHECK, Render.com uses / from render.yaml
# Note: PORT env var is set by Render.com, health check uses default 80 if not set
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD sh -c 'PORT=${PORT:-80}; wget --no-verbose --tries=1 --spider http://localhost:$PORT/health 2>/dev/null || exit 1'

# Use custom entrypoint that handles PORT default, then calls nginx:alpine's entrypoint
ENTRYPOINT ["/docker-entrypoint-custom.sh"]
CMD ["nginx", "-g", "daemon off;"]

