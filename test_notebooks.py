#!/usr/bin/env python3
"""Script para ejecutar todos los notebooks y verificar errores"""

import subprocess
import sys
from pathlib import Path

def test_notebook(notebook_path):
    """Ejecuta un notebook y retorna True si tiene éxito"""
    try:
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", 
             "--inplace", str(notebook_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout al ejecutar el notebook"
    except Exception as e:
        return False, str(e)

def main():
    """Ejecuta todos los notebooks en el proyecto"""
    base_dir = Path(".")
    notebooks = []
    
    # Buscar todos los notebooks
    for notebook_dir in ["00-basicos", "01-python", "02-numpy", "03-pandas", "04-matplotlib", "05-scikit-learn"]:
        dir_path = base_dir / notebook_dir
        if dir_path.exists():
            notebooks.extend(dir_path.glob("*.ipynb"))
    
    notebooks.sort()
    
    print(f"📚 Encontrados {len(notebooks)} notebooks\n")
    
    errors = []
    for notebook in notebooks:
        print(f"🔄 Ejecutando {notebook}...", end=" ")
        success, error = test_notebook(notebook)
        if success:
            print("✅ OK")
        else:
            print("❌ ERROR")
            errors.append((notebook, error))
            if error:
                error_lines = error.split('\n')
                for line in error_lines[-15:]:  # Últimas 15 líneas del error
                    if line.strip() and ('Error' in line or 'Traceback' in line or 'Exception' in line or 'ModuleNotFoundError' in line):
                        print(f"   {line[:150]}")
    
    print("\n" + "="*60)
    if errors:
        print(f"❌ {len(errors)} notebook(s) con errores:")
        for notebook, error in errors:
            print(f"  - {notebook}")
        return 1
    else:
        print("✅ Todos los notebooks se ejecutaron correctamente!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

