#!/usr/bin/env python3
"""Script para ejecutar todos los notebooks y detectar errores"""

import subprocess
import sys
import os
from pathlib import Path

def find_notebooks(root_dir):
    """Encuentra todos los notebooks .ipynb"""
    notebooks = []
    for path in Path(root_dir).rglob("*.ipynb"):
        notebooks.append(str(path))
    return sorted(notebooks)

def execute_notebook(notebook_path):
    """Ejecuta un notebook y retorna el resultado"""
    try:
        result = subprocess.run(
            ["python3", "-m", "jupyter", "nbconvert", 
             "--to", "notebook", "--execute", "--inplace", notebook_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos máximo por notebook
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout después de 5 minutos"
    except Exception as e:
        return False, "", str(e)

def main():
    root_dir = "."
    notebooks = find_notebooks(root_dir)
    
    print(f"Encontrados {len(notebooks)} notebooks\n")
    
    errors = []
    success = []
    
    for i, notebook in enumerate(notebooks, 1):
        print(f"[{i}/{len(notebooks)}] Ejecutando: {notebook}...", end=" ", flush=True)
        success_flag, stdout, stderr = execute_notebook(notebook)
        
        if success_flag:
            print("✓ OK")
            success.append(notebook)
        else:
            print("✗ ERROR")
            errors.append((notebook, stdout, stderr))
    
    print(f"\n{'='*60}")
    print(f"Resumen:")
    print(f"  Exitosos: {len(success)}")
    print(f"  Con errores: {len(errors)}")
    
    if errors:
        print(f"\n{'='*60}")
        print("Notebooks con errores:")
        for notebook, stdout, stderr in errors:
            print(f"\n{notebook}:")
            if stderr:
                print(f"  STDERR: {stderr[:500]}")
            if stdout and "error" in stdout.lower():
                print(f"  STDOUT: {stdout[:500]}")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






