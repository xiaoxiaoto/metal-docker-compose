FROM nginx:alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY router.mjs /etc/nginx/router.mjs
