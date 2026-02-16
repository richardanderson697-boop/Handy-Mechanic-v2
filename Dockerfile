# syntax=docker/dockerfile:1
FROM node:18-slim

WORKDIR /app

# 1. Copy package files first
COPY package.json ./

# 2. Force install ignoring all audits and lockfile issues

RUN npm install --legacy-peer-deps

# 3. Copy the rest of the code from the current main folder
COPY . .

# 4. Diagnostic: List files to ensure 'app' and 'public' are visible
RUN ls -la

# 5. Build the Next.js app
RUN npm run build

# 6. Set Railway standard port
ENV PORT=3000
EXPOSE 3000

CMD ["npm", "start"]
