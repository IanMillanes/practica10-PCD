#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Cargar datos
titanic_df = pd.read_csv('titanic.csv')

# Configuración del gráfico
plt.figure(figsize=(12, 6))
plt.title('Distribución de Edades en el Titanic', fontsize=14, pad=20)

# Histograma con bins personalizados y KDE
sns.histplot(data=titanic_df, 
             x='Age', 
             bins=np.arange(0, 81, 5),  # Bins cada 5 años desde 0 a 80
             kde=True,                  # Línea de densidad suavizada
             color='#4e79a7',           # Color profesional
             edgecolor='white',         # Bordes blancos para definición
             alpha=0.7)                 # Transparencia para mejor visualización

# Personalización de ejes
plt.xlabel('Edad (años)', fontsize=12)
plt.ylabel('Número de Pasajeros', fontsize=12)
plt.xticks(np.arange(0, 81, 10))       # Marcas cada 10 años
plt.grid(axis='y', linestyle='--', alpha=0.4)  # Grid horizontal suave

plt.tight_layout()
plt.show()
# %%

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar y preparar datos
titanic_df = pd.read_csv('titanic.csv')

# Configuración del gráfico
plt.figure(figsize=(10, 6))
plt.title('Distribución de Edades por Clase y Supervivencia', fontsize=14, pad=20)

# Boxplot agrupado
sns.boxplot(
    x='Pclass',              # Eje X: Clase de pasaje
    y='Age',                 # Eje Y: Edad
    hue='Survived',          # Color por supervivencia
    data=titanic_df,         # Datos
    palette={0: '#e74c3c',   # Rojo para no sobrevivientes
             1: '#2ecc71'},  # Verde para sobrevivientes
    showmeans=True,          # Mostrar media (triángulo)
    meanprops={'marker':'^', 'markerfacecolor':'white'}  # Estilo de la media
)

# Personalización
plt.xlabel('Clase de Pasaje', fontsize=12)
plt.ylabel('Edad (años)', fontsize=12)
plt.xticks([0, 1, 2], ['Primera', 'Segunda', 'Tercera'])
plt.legend(title='Sobrevivió', labels=['No', 'Sí'])
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


#%%
import matplotlib.pyplot as plt
import pandas as pd

# Cargar y preparar datos
titanic_df = pd.read_csv('titanic.csv').dropna(subset=['Age'])
titanic_df['Familiares'] = titanic_df['SibSp'] + titanic_df['Parch']

# Configuración simplificada
plt.figure(figsize=(12, 7))

# Scatter plot básico con colores manuales
colors = {1: 'gold', 2: 'silver', 3: 'brown'}
plt.scatter(
    x=titanic_df['Age'],
    y=titanic_df['Familiares'],
    c=titanic_df['Pclass'].map(colors),  # Mapeo directo de colores
    s=titanic_df['Familiares']*10 + 20,  # Tamaño proporcional
    alpha=0.7
)

# Personalización
plt.title('Edad vs. Familiares a Bordo', fontsize=16)
plt.xlabel('Edad (años)', fontsize=12)
plt.ylabel('Número de Familiares', fontsize=12)
plt.xticks(range(0, 81, 10))
plt.grid(ls='--', alpha=0.3)

# Leyenda simplificada
for class_num, color in colors.items():
    plt.scatter([], [], c=color, label=f'Clase {class_num}', alpha=0.7)
plt.legend(title='Clase', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
# %%




#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
img = Image.open('titanic.jpg')
img_array = np.array(img)
plt.grid(False)
plt.title('Una foto real del Titanic')
plt.imshow(img)
# %%


#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image

# Cargar datos
titanic_df = pd.read_csv('titanic.csv')
img = Image.open('titanic.jpg')
img_array = np.array(img)

# Crear figura con subplots (2 filas, 2 columnas)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Análisis Visual del Titanic', fontsize=20, y=1.02)

# --- Gráfico 1: Histograma de edades ---
sns.histplot(ax=axes[0,0], 
             data=titanic_df, 
             x='Age', 
             bins=np.arange(0, 81, 5),
             kde=True,
             color='#4e79a7',
             edgecolor='white',
             alpha=0.7)
axes[0,0].set_title('Distribución de Edades', fontsize=14)
axes[0,0].set_xlabel('Edad (años)', fontsize=12)
axes[0,0].set_ylabel('Número de Pasajeros', fontsize=12)
axes[0,0].grid(axis='y', linestyle='--', alpha=0.4)

# --- Gráfico 2: Boxplot edades por clase/supervivencia ---
sns.boxplot(ax=axes[0,1],
            x='Pclass',
            y='Age',
            hue='Survived',
            data=titanic_df,
            palette={0: '#e74c3c', 1: '#2ecc71'},
            showmeans=True,
            meanprops={'marker':'^', 'markerfacecolor':'white'})
axes[0,1].set_title('Edad por Clase y Supervivencia', fontsize=14)
axes[0,1].set_xlabel('Clase de Pasaje', fontsize=12)
axes[0,1].set_ylabel('Edad (años)', fontsize=12)
axes[0,1].set_xticklabels(['Primera', 'Segunda', 'Tercera'])
axes[0,1].legend(title='Sobrevivió', labels=['No', 'Sí'])
axes[0,1].grid(axis='y', linestyle='--', alpha=0.3)

# --- Gráfico 3: Scatter plot edad vs familiares ---
colors = {1: 'gold', 2: 'silver', 3: 'brown'}
titanic_clean = titanic_df.dropna(subset=['Age', 'SibSp', 'Parch'])
titanic_clean['Familiares'] = titanic_clean['SibSp'] + titanic_clean['Parch']

sc = axes[1,0].scatter(
    x=titanic_clean['Age'],
    y=titanic_clean['Familiares'],
    c=titanic_clean['Pclass'].map(colors),
    s=titanic_clean['Familiares']*10 + 20,
    alpha=0.7
)
axes[1,0].set_title('Edad vs. Familiares a Bordo', fontsize=14)
axes[1,0].set_xlabel('Edad (años)', fontsize=12)
axes[1,0].set_ylabel('Número de Familiares', fontsize=12)
axes[1,0].grid(linestyle='--', alpha=0.3)

# Leyenda manual para el scatter plot
for class_num, color in colors.items():
    axes[1,0].scatter([], [], c=color, label=f'Clase {class_num}', alpha=0.7)
axes[1,0].legend(title='Clase', bbox_to_anchor=(1.05, 1))

# --- Gráfico 4: Imagen del Titanic ---
axes[1,1].imshow(img_array)
axes[1,1].set_title('Foto Real del Titanic', fontsize=14)
axes[1,1].grid(False)
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])

# Ajustar espacio entre gráficos
plt.tight_layout(pad=3)
plt.show()
# %%
