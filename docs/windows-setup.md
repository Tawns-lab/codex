# Windows Development Environment Setup Guide

This guide will help you set up a complete development environment on Windows for working with Codex and C/C++ development.

## Overview

While **WSL2 (Windows Subsystem for Linux) is the officially recommended approach** for running Codex on Windows, this guide covers both options:

1. **WSL2 Setup (Recommended)** - Full Linux environment on Windows
2. **Native Windows Setup** - Using Git Bash and MinGW

## Option 1: WSL2 Setup (Recommended)

WSL2 provides a complete Linux environment on Windows and is the officially supported method.

### Prerequisites

- Windows 11 or Windows 10 version 2004 or higher (Build 19041 or higher)
- Administrator access

### Installation Steps

#### 1. Install WSL2

Open PowerShell or Command Prompt as **Administrator** and run:

```powershell
wsl --install
```

This will:
- Enable WSL and Virtual Machine Platform
- Download and install the latest Linux kernel
- Install Ubuntu as the default Linux distribution

Restart your computer when prompted.

#### 2. Set Up Ubuntu

After restart, Ubuntu will automatically launch and ask you to create a username and password.

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y build-essential git curl
```

#### 3. Install Node.js (for Codex)

```bash
# Install Node.js 22 (LTS)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version  # Should show v22.x.x
npm --version
```

#### 4. Install Codex

```bash
npm install -g @openai/codex

# Verify installation
codex --version
```

#### 5. Set Up API Key

```bash
# Add to your ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### 6. Test Your Setup

```bash
# Test GCC
gcc --version

# Test Codex
codex --help

# Create a test C program
cat > hello.c << 'EOF'
#include <stdio.h>

int main() {
    printf("Hello from WSL2!\n");
    return 0;
}
EOF

# Compile and run
gcc hello.c -o hello
./hello
```

### Accessing Windows Files from WSL2

Your Windows drives are mounted at `/mnt/`:
- C: drive → `/mnt/c/`
- D: drive → `/mnt/d/`

```bash
# Navigate to your Windows user folder
cd /mnt/c/Users/YourUsername/

# Or create projects in WSL's home directory
cd ~
mkdir projects
```

### Using VS Code with WSL2

1. Install [Visual Studio Code](https://code.visualstudio.com/) on Windows
2. Install the "WSL" extension in VS Code
3. In WSL2 terminal, navigate to your project and run:
   ```bash
   code .
   ```

## Option 2: Native Windows Setup

This setup uses Git Bash and MinGW for a native Windows development environment.

> **Note:** Native Windows support is not officially supported but may work. Use WSL2 for the best experience.

### Prerequisites

- Windows 10 or 11
- Administrator access (for some installations)

### Installation Steps

#### 1. Install Git for Windows (includes Git Bash)

1. Download Git for Windows from: https://git-scm.com/downloads/win
2. Run the installer with these recommended settings:
   - ✅ Use Git from Git Bash only (or from the command line)
   - ✅ Use the OpenSSL library
   - ✅ Checkout as-is, commit Unix-style line endings
   - ✅ Use MinTTY (default terminal of Git Bash)
   - ✅ Enable file system caching

3. Verify installation:
   ```bash
   # Open Git Bash and run:
   git --version
   bash --version
   ```

#### 2. Install MinGW-w64 (GCC for Windows)

**Option A: Using MSYS2 (Recommended)**

1. Download MSYS2 from: https://www.msys2.org/
2. Run the installer and follow the prompts
3. After installation, open "MSYS2 MSYS" from the Start Menu
4. Update the package database:
   ```bash
   pacman -Syu
   ```
5. Close and reopen MSYS2, then install development tools:
   ```bash
   pacman -S --needed base-devel mingw-w64-x86_64-toolchain
   ```
6. Add to your Windows PATH:
   - Open System Properties → Environment Variables
   - Edit the `Path` variable for your user
   - Add: `C:\msys64\mingw64\bin`
   - Add: `C:\msys64\usr\bin`

**Option B: Using winget (Windows Package Manager)**

If you have Windows 11 or Windows 10 with winget:

```powershell
# Open PowerShell and run:
winget install -e --id MSYS2.MSYS2
```

Then follow steps 3-6 from Option A.

#### 3. Install Node.js

1. Download Node.js 22 LTS from: https://nodejs.org/
2. Run the installer (accept defaults)
3. Verify in Command Prompt or Git Bash:
   ```bash
   node --version  # Should show v22.x.x
   npm --version
   ```

4. Update npm to latest version:
   ```bash
   npm install -g npm@latest
   ```

#### 4. Install Codex

Open Git Bash and run:

```bash
npm install -g @openai/codex
```

#### 5. Configure Git Bash Path for Codex (if needed)

If you get an error about git-bash when running `codex`, set the environment variable:

**Via Command Prompt:**
```cmd
setx CLAUDE_CODE_GIT_BASH_PATH "C:\Program Files\Git\bin\bash.exe"
```

**Via PowerShell:**
```powershell
[Environment]::SetEnvironmentVariable("CLAUDE_CODE_GIT_BASH_PATH", "C:\Program Files\Git\bin\bash.exe", "User")
```

**Permanently (System-wide):**
1. Open System Properties → Environment Variables
2. Click "New" under User variables
3. Variable name: `CLAUDE_CODE_GIT_BASH_PATH`
4. Variable value: `C:\Program Files\Git\bin\bash.exe`
5. Click OK

#### 6. Set Up API Key

**Via Command Prompt (current session only):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Via PowerShell (current session only):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Permanently:**
```cmd
setx OPENAI_API_KEY "your-api-key-here"
```

**Using .env file (recommended for projects):**

Create a `.env` file in your project directory:
```
OPENAI_API_KEY=your-api-key-here
```

#### 7. Test Your Setup

Open Git Bash:

```bash
# Test GCC
gcc --version

# Test Node.js
node --version

# Test Codex
codex --version

# Create a test C program
cat > hello.c << 'EOF'
#include <stdio.h>

int main() {
    printf("Hello from Windows!\n");
    return 0;
}
EOF

# Compile and run
gcc hello.c -o hello.exe
./hello.exe
```

## Common Issues and Solutions

### Issue: `gcc` not recognized

**Solution:**
- Verify MinGW is installed: `C:\msys64\mingw64\bin\gcc.exe` should exist
- Check your PATH includes: `C:\msys64\mingw64\bin`
- Restart your terminal/command prompt after PATH changes
- Open a new Git Bash window

### Issue: `claude` or `codex` not recognized

**Solution:**
- Run `npm install -g @openai/codex` (note: correct package name)
- Check npm global bin directory is in PATH:
  ```bash
  npm config get prefix
  ```
- The bin folder should be in your PATH (usually `C:\Users\YourName\AppData\Roaming\npm`)

### Issue: Permission denied when running executables

**Solution in Git Bash:**
```bash
# Use explicit path
./hello.exe

# Or add execute permission (if using MSYS2)
chmod +x hello.exe
./hello.exe
```

**Solution in Command Prompt:**
```cmd
# Just run without ./
hello.exe
```

### Issue: Line ending problems (CRLF vs LF)

**Solution:**
Configure Git to handle line endings properly:
```bash
git config --global core.autocrlf true
```

### Issue: Codex requires git-bash

**Solution:**
Set the environment variable pointing to bash.exe:
```cmd
setx CLAUDE_CODE_GIT_BASH_PATH "C:\Program Files\Git\bin\bash.exe"
```

Then restart your terminal.

## Recommended Development Workflow

### For WSL2 Users:
1. Keep your code in WSL2 filesystem (`~/projects/`) for better performance
2. Use VS Code with WSL extension
3. Run all development tools from WSL2 terminal

### For Native Windows Users:
1. Use Git Bash for command-line operations
2. Keep your code in Windows filesystem (`C:\Users\YourName\projects\`)
3. Use a code editor with good Windows support (VS Code, Sublime Text, etc.)
4. Be mindful of line ending differences

## Next Steps

1. **Configure Codex:** See [config.md](./config.md) for configuration options
2. **Learn Authentication:** See [authentication.md](./authentication.md) for API key setup
3. **Explore Features:** See [getting-started.md](./getting-started.md) for usage examples

## Additional Resources

- [WSL2 Documentation](https://learn.microsoft.com/en-us/windows/wsl/)
- [Git for Windows](https://git-scm.com/download/win)
- [MSYS2 Documentation](https://www.msys2.org/)
- [Node.js Downloads](https://nodejs.org/)
- [Codex GitHub Repository](https://github.com/openai/codex)

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review the [FAQ](./faq.md)
3. Search [GitHub Issues](https://github.com/openai/codex/issues)
4. Open a new issue with:
   - Your Windows version
   - Your setup choice (WSL2 or Native)
   - Error messages
   - Steps to reproduce

---

**Tip:** For the most reliable experience on Windows, use WSL2. It provides a complete Linux environment and is fully supported.
