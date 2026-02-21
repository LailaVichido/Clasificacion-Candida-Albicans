%% ========================================================================
%  1. CONFIGURACIÓN CENTRALIZADA
%  ========================================================================
% Usamos una 'struct' para mantener todo ordenado y fácil de ajustar
params = struct();

% -- Preprocesamiento
params.preprocess.medfilt_kernel = [5 5]; % Kernel de filtro mediano

% -- Segmentación
params.segment.sensitivity = 0.38; % Sensibilidad de binarización
params.segment.watershed_minima_suppression = 2; % Controla la sobre-segmentación de watershed. >1 previene divisiones innecesarias.

% -- Filtrado de Objetos por Propiedades
params.filter.min_area = 500;    % Área mínima en píxeles
params.filter.max_area = 8000;  % Área máxima
params.filter.min_solidity = 0.6; % Descarta objetos muy irregulares/ruidosos

% -- Clasificación
% Rangos para normalizar [0, 1]
params.class.circ_range = [0.25 1.0]; % Circularity
params.class.sol_range  = [0.80 1.0]; % Solidity
params.class.ecc_range  = [0.40 0.99]; % Eccentricity (0=círculo, 1=línea)
% Pesos para los scores
params.class.blasto_weights = [0.5 0.5]; % [Peso_Compacidad, Peso_Redondez]
params.class.hifa_weights = 0.7;         % [Peso_Alargamiento]
% Decisión
params.class.decision_margin = 0.08; % Margen mínimo para no ser "Incierto"

%% ========================================================================
%  2. CARGA Y PREPROCESAMIENTO
%  ========================================================================
img_orig = imread('Imagen 3.tif');
g = rgb2gray(img_orig);

% Mejorar contraste y aplicar filtro mediano
Iadj = imadjust(g);
Ifilt = medfilt2(Iadj, params.preprocess.medfilt_kernel);

%% ========================================================================
%  3. SEGMENTACIÓN AVANZADA (ADAPTIVE + WATERSHED)
%  ========================================================================
% Paso 1: Binarización inicial para obtener la máscara principal
BW = imbinarize(Ifilt, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', params.segment.sensitivity);
BW = ~BW; % Objetos en blanco

% Paso 2: Limpieza morfológica inicial
BW_clean = bwareaopen(BW, params.filter.min_area);

% Paso 3: Transformada de distancia (preparación para Watershed)
% El valor en cada píxel será la distancia al fondo más cercano.
% Los centros de los objetos serán picos.
D = -bwdist(~BW_clean);

% Paso 4: Supresión de mínimos para evitar sobre-segmentación
% Evita que un solo objeto se divida en muchos pedazos.
mask = imextendedmin(D, params.segment.watershed_minima_suppression);
D2 = imimposemin(D, mask);

% Paso 5: Algoritmo de Watershed (el chido)
% L es una matriz de etiquetas. Cada objeto tiene un número entero único.
L = watershed(D2);
L(BW_clean == 0) = 0; % Asegurar que el fondo sea 0

%% ========================================================================
%  4. EXTRACCIÓN DE CARACTERÍSTICAS Y FILTRADO
%  ========================================================================
% regionprops ahora funciona sobre la matriz de etiquetas 'L'
S = regionprops(L, 'Area', 'Solidity', 'Circularity', ...
                   'Eccentricity', 'Centroid', 'PixelIdxList');

% Filtrado robusto basado en las propiedades de la struct de parámetros
keep_idx = find([S.Area] >= params.filter.min_area & ...
                [S.Area] <= params.filter.max_area & ...
                [S.Solidity] >= params.filter.min_solidity);
S = S(keep_idx);

% Si no queda nada, terminar
if isempty(S)
    disp('No se encontraron objetos que cumplan los criterios.');
    return;
end

% Extraer vectores de características
Circ = [S.Circularity]';
Sol  = [S.Solidity]';
Ecc  = [S.Eccentricity]';
Cen  = vertcat(S.Centroid);

%% ========================================================================
%  5. CLASIFICACIÓN REFINADA
%  ========================================================================
% Función de normalización
norm01 = @(x, range) (max(min(x, range(2)), range(1)) - range(1)) / max(diff(range), eps);

% Normalizar características
CIRn = norm01(Circ, params.class.circ_range);
SOLn = norm01(Sol,  params.class.sol_range);
ECCn = norm01(Ecc,  params.class.ecc_range);

% -- Scores más intuitivos --
% Score de Compacidad/Redondez (para Blastos)
score_compact = params.class.blasto_weights(1) * SOLn + ...
                params.class.blasto_weights(2) * CIRn;

% Score de Alargamiento (para Hifas)
score_elongated = params.class.hifa_weights * ECCn;

% Decisión
score_blasto = score_compact - 0.2*score_elongated; % Penalizar si es alargado
score_hifa   = score_elongated - 0.3*score_compact; % Penalizar si es compacto

margin = abs(score_blasto - score_hifa);
is_hifa = score_hifa > score_blasto;

label = strings(numel(S), 1);
label(is_hifa & margin >= params.class.decision_margin) = "PSEUDOHIFA";
label(~is_hifa & margin >= params.class.decision_margin) = "BLASTO";
label(margin < params.class.decision_margin) = "INCIERTO";

% Conteo
fprintf('--- Conteo Final ---\n');
fprintf('Total: %d | Pseudohifas: %d | Blastos: %d | Inciertos: %d\n', ...
        numel(label), sum(label=="PSEUDOHIFA"), sum(label=="BLASTO"), sum(label=="INCIERTO"));

%% ========================================================================
%  6. VISUALIZACIÓN
%  ========================================================================
% Crear una nueva matriz de etiquetas solo con los objetos clasificados
final_labels = zeros(size(g));
for i = 1:numel(S)
    % Asignar un código numérico a cada clase (1: Hifa, 2: Blasto, 3: Incierto)
    class_code = 1*strcmp(label(i),"PSEUDOHIFA") + ...
                 2*strcmp(label(i),"BLASTO") + ...
                 3*strcmp(label(i),"INCIERTO");
    final_labels(S(i).PixelIdxList) = class_code;
end

% Paleta de colores PRO
colors = [0 0.8 1;   % 1: PSEUDOHIFA (Cyan)
          1 0 0.6;   % 2: BLASTO (Magenta)
          1 0.85 0]; % 3: INCIERTO (Amarillo)

% Crear una imagen RGB con las máscaras de los objetos coloreadas
rgb_overlay = label2rgb(final_labels, colors, 'k'); % 'k' para fondo negro

% Mostrar la imagen original y superponer las máscaras con transparencia
figure('Color', 'k');
imshow(img_orig);
hold on;
% La magia sucede aquí: se superpone la capa de color con 60% de opacidad
hImg = imshow(rgb_overlay);
set(hImg, 'AlphaData', 0.6);

% --- Leyenda personalizada y profesional ---
hold on;
h1 = plot(nan,nan,'s','MarkerSize',15,'MarkerFaceColor',colors(1,:),'MarkerEdgeColor','w');
h2 = plot(nan,nan,'s','MarkerSize',15,'MarkerFaceColor',colors(2,:),'MarkerEdgeColor','w');
h3 = plot(nan,nan,'s','MarkerSize',15,'MarkerFaceColor',colors(3,:),'MarkerEdgeColor','w');
legend([h1 h2 h3], {'Pseudohifa', 'Blasto', 'Incierto'}, ...
       'TextColor','w','EdgeColor',[.5 .5 .5],'Color',[.1 .1 .1],...
       'Location','southoutside','Orientation','horizontal', 'FontSize', 12);
title('Clasificación de Candida Albicans', 'Color', 'w', 'FontSize', 16);
hold off;