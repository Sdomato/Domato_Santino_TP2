## **Dataset: Diagnóstico de Cáncer de Mama**
E
El propósito es evaluar si una célula presenta características compatibles con unaste conjunto de datos proviene de **tres laboratorios de investigación biomédica**, que han recolectado información sobre muestras celulares. condición médica, basada en mediciones morfológicas y bioquímicas de las células obtenidas en estudios clínicos.  
---
## **Descripción de las Variables**

### **Variables Numéricas (mediciones de laboratorio)**
- **CellSize** *(µm)* → Tamaño de la célula en micrómetros. 
- **CellShape** *(ratio)* → Relación de aspecto de la célula. 
- **NucleusDensity** *(g/cm³)* → Densidad del núcleo celular. 
- **ChromatinTexture** *(unidades arbitrarias)* → Textura de la cromatina. 
- **CytoplasmSize** *(µm)* → Tamaño del citoplasma.
- **CellAdhesion** *(0-1)* → Capacidad de adhesión celular. *(0: baja, 1: alta)*  
- **MitosisRate** *(eventos/unidad de tiempo)* → Frecuencia de mitosis observada.
- **NuclearMembrane** *(1-5)* → Estado de la membrana nuclear *(1: frágil, 5: íntegra)*.  
- **GrowthFactor** *(ng/mL)* → Concentración de factores de crecimiento.
- **OxygenSaturation** *(%)* → Nivel de oxigenación de la célula. 
- **Vascularization** *(0-10)* → Índice de vascularización celular. 
- **InflammationMarkers** *(0-100)* → Nivel de marcadores inflamatorios.

### **Variables Categóricas**
- **CellType** → Clasificación del tipo celular. *(`Epithelial`, `Mesenchymal`, `Unknown`)*  
- **GeneticMutation** → Presencia de mutaciones genéticas. *(`Present`, `Absent`, `Unknown`)*  

### **Variable Objetivo**
- **Diagnosis** *(0/1)* → Indica si la célula presenta características compatibles con la condición médica estudiada.  
  - `0` → Célula sin anomalías.  
  - `1` → Célula con características anómalas.  

  \subsection{Introducción}
El problema abordado en este trabajo consiste en el diagnóstico de cáncer de mama a partir de datos celulares obtenidos en estudios clínicos. La entrada del algoritmo son mediciones morfológicas y bioquímicas de las células, las cuales se procesaron para extraer características relevantes para el diagnóstico. Debido a que el dataset original se encontraba desbalanceado, se aplicaron distintos métodos de re-balanceo para ajustar la distribución de las clases.

En este estudio se utilizaron únicamente modelos de regresión logística, entrenados con cada uno de los métodos de re-balanceo propuestos, para realizar una clasificación binaria. El objetivo del algoritmo es, a partir de las mediciones celulares, predecir si el paciente presenta características compatibles con la condición médica estudiada o no, lo cual se evalúa mediante métricas de desempeño como accuracy, precision, recall y F1-score.
