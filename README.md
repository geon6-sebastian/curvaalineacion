# curvas.py

## Tabla de Contenidos

- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Ejemplos](#-ejemplos)
- [Licencia](#-licencia)

## Requisitos

- **Python 3.x**

```bash
pip install numba
pip install pandas
pip install geopandas pyshp simplekml shapely
```

## Instalación

**Clonar el repositorio:**
```bash
git clone https://github.com/geon6-sebastian/curvaalineacion.git
cd curvaalineacion
```
---

## Uso

Para ejecutar el script, utiliza el siguiente comando en la terminal:

```bash
python curvas.py [argumentos]
```
### Argumentos


| Argumento                                  | Descripción                                                                                                         | Requerido        | Default                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------- |
| -i, --inverso                              | Ejecutar problema inverso                                                                                            | No               | -                          |
| -d, --directo                              | Ejecutar problema directo                                                                                            | No               | -                          |
| -poly, --poly-sup ('coords.csv')           | Calcula la superficie dentro de un polígono dado en un archivo CSV/TXT                                              | No               | -                          |
| -t, --tipo ['align', 'normal', 'central'] |                                                                                                                      | No               | 'central'                  |
| -P1 (latitud longitud)                     | Punto 1: latitud longitud (en grados decimales). Requerido para -i y -d                                              | Si, para -i y -d | -                          |
| -P2 (latitud longitud)                     | Punto 1: latitud longitud (en grados decimales). Requerido para -i                                                   | Si, para -i      | -                          |
| -e (a, inv_f)                              | Elipsoide: semieje_mayor inversa_aplastamiento (por defecto: GRS80)                                                  |                  | GRS80_a, 298.2572221008827 |
| -o, --output ('nombrearchivo')             | Nombre base para guardar salidas (KMZ, SHP, CSV). Este comando SOBREESCRIBE los archivos existentes del mismo nombre | No               | -                          |
| -az (acimut)                               | Acimut inicial (en grados decimales). Requerido para -d                                                              | Si, para -d      | -                          |
| -s (distancia)                             | Distancia (en metros). Requerido para -d                                                                             | Si, para -d      | -                          |
| -mstep, --max-step (paso)                  | Paso máximo de h para Dormand-Prince en grados decimales                                                            | No               | 0.1                        |

---

## Ejemplos

**Ejemplo básico (Problema directo, paso 0.01 grados):**

```bash
python curvas.py -d -P1 -30 -60 -a 30 -s 5000000 -t align -o align_0.01 -mstep 0.01
```
**Cálculo de la superficie de un polígono uniendo vértices con la curva de alineación:**

```bash
python curvas.py -poly coords.csv -o nombre_poligono -t align
```
---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---
