# syntax=docker/dockerfile:1
FROM node:18-slim
WORKDIR /app
COPY package.json ./
RUN npm install --legacy-peer-deps
COPY . .
RUN npm run build
ENV PORT=3000
EXPOSE 3000
CMD ["npm", "start"]
