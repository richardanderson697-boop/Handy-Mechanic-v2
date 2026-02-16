# syntax=docker/dockerfile:1
FROM node:18-slim

WORKDIR /app

# 1. Copy only the package file first to cache the install step
COPY package.json ./

# 2. Force install ignoring all audits and lockfiles
RUN npm install --no-audit --fund=false --force

# 3. Copy EVERYTHING else from your main directory
COPY . .

# 4. Debug: List files so we can see them in the logs if it fails
RUN ls -la

# 5. Build the Next.js app
RUN npm run build

# 6. Railway needs to know which port to open
ENV PORT 3000
EXPOSE 3000

CMD ["npm", "start"]
