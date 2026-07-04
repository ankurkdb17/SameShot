# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Build
# Compile the Vite/React app to static assets.
# ──────────────────────────────────────────────────────────────────────────────
FROM node:22-slim AS build

WORKDIR /app

# Leverage Docker layer caching: install deps before copying source.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# VITE_API_BASE_URL is baked into the static build at build time (Vite env
# vars are compile-time, not runtime) — override via --build-arg if the
# backend isn't at the default location.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Serve
# Lean nginx image serving the pre-built static files.
# ──────────────────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html

# React Router uses client-side routing — unknown paths must fall back to
# index.html rather than returning nginx's default 404.
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    location / {\n\
        try_files $uri /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
