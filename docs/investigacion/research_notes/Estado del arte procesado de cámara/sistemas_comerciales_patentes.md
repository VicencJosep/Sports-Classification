# Sistemas comerciales de cámara automática para deporte amateur y sus patentes y publicaciones técnicas

Notas de investigación web. Todas las páginas se consultaron el 2026-11-17 salvo que se indique la fecha de publicación.
Convenciones: **[F]** = lo afirma el fabricante; **[C]** = lo afirma un competidor (sesgo comercial); **[T]** = tercero
independiente; **«sin verificar»** = dato visto solo en el resumen de un buscador o en una página que no pude abrir
(403/404/429 o PDF ilegible), no citar como hecho. No hay nada rellenado de memoria.

Páginas que NO pude abrir y que contienen datos relevantes: centro de ayuda de Veo (`support.veo.com`, 403), fichas de
minoristas de Pixellot Air NXT (B&H, Videoguys, AVLGear: 403/429), Justia Patents (403), el PDF de Cabado et al. en CVF
(descargado, pero sin extractor de PDF en el host), el PDF de la FCF sobre Veo, y la ayuda de Reeplayer (404).

## 1. Hardware, captura, vídeo entregado, dónde se hace el seguimiento y si se puede bajar la panorámica (sistema por sistema)

### Takeaway
Hay tres familias: (a) **panorámica fija multi-sensor cosida + cámara virtual por recorte** (Veo, Pixellot, Spiideo, Hudl
Focus, BePro, Zone14, AutomaticTV, Reeplayer, Trace, Once Autocam), que graban todo el campo y entregan además una vista
«follow-cam» recortada, normalmente a 1080p; (b) **cabezal motorizado que apunta una cámara** (XbotGo, Pix4Team), que solo
graban lo que encuadran; y (c) **software sobre cámaras de terceros** (Studio Automated, Once Autocam). Casi ningún
fabricante publica bitrate, códec ni resolución exacta de la panorámica descargable; el dato duro más repetido es
«captura 4K por sensor, vista producida 1080p».

### Cited Findings

**Veo (Dinamarca) — Veo Cam 3, Cam 3 5G, Veo Go; antes Cam 1 y Cam 2**
- [F] La follow-cam «is created from a panoramic video stitched together from two synchronized streams… Our AI then analyses the footage, recognizes where the game is happening, and cuts out exactly that area to deliver a 1080p follow-cam view» — [Veo Cam 3](https://www.veo.com/en-us/product/veo-cam-3)
- [F] Cobertura de 180° con «Interactive Mode» (el usuario panea y hace zoom sobre el campo completo); 1,25 kg; batería «up to 6.5 hours»; funciona de −10 a 45 °C con lluvia o nieve; «SteadyView is automatically applied after every recording» (estabilización posterior); trípodes de 17,1 ft y 24,3 ft (≈5,2 m y ≈7,4 m) — [Veo Cam 3](https://www.veo.com/en-us/product/veo-cam-3)
- [F] «Veo cameras utilize dual 4K lenses to record the entire pitch/field (180° view) continuously… the full panoramic source file is always preserved» — [veo.com/llms.txt](https://www.veo.com/llms.txt)
- [F] Veo Cam 3 «4K capture with onboard AI that follows the action automatically»; la Cam 3 5G añade módem 5G; Veo Go usa el iPhone — [Veo pricing](https://www.veo.com/en-us/pricing)
- **Conflicto de fps**: la comparativa propia de Veo de 2026 dice «4K at 60fps… Veo Cam 3 also records at 4K 60fps compared to Pixellot's 30fps» — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared); el artículo de soporte «Compare Veo Cam 2, Veo Cam 3…» figura en el buscador con «4K resolution at 30 fps» para Cam 2, Cam 3 y Cam 3 5G (**sin verificar**, 403: `https://support.veo.com/hc/en-us/articles/22689559243537`).
- **Dónde se hace el seguimiento**: mensajes mezclados. [F] «generates a tracked follow-cam view after upload» — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared); pero también «the follow-cam is generated the instant you hit record» — [Veo Cam 3](https://www.veo.com/en-us/product/veo-cam-3) y «onboard AI» — [Veo pricing](https://www.veo.com/en-us/pricing). En 2017 la IA corría en la nube: «takes advantage of compute resources in the cloud to follow the action» — [NVIDIA blog, 2017-08-31](https://developer.nvidia.com/blog/ai-technology-automatically-records-soccer-matches)
- [F] En el Veo Editor hay dos vistas: follow-cam e «interactive mode… or switch to panoramic mode for a complete pitch overview» — [Veo Editor](https://www.veo.com/product/veo-editor)
- [F] Descarga: «you can download your Veo Cam 3 video recordings directly from the Veo Editor»; clips en MP4 — [Veo Cam 3](https://www.veo.com/en-us/product/veo-cam-3); «unlimited downloads and sharing» — [Veo vs Trace](https://www.veo.com/en-us/veo-vs-trace)
- [F] Veo Go: dos iPhone (11 o posterior, pueden ser modelos distintos) en un soporte; «Our AI merges the footage into a panoramic field view with a smooth follow-cam»; ~6 GB por hora y por iPhone; «Download full games» — [Veo Go](https://www.veo.com/en-us/product/veo-go)
- [F] Planes por cámara: Starter 10 h de grabación/mes, 1 equipo, 15 usuarios; Team 20 h, 1 equipo, 30 usuarios; Club 200 h, 10 equipos, 300 usuarios; Enterprise 600 h, 20 equipos, 600 usuarios — [Veo pricing](https://www.veo.com/en-us/pricing)
- [T] Primera generación (2018-2020): dos lentes, 180°, «4K resolution at 30 frames per second», 1.499 $; más de 2.500 clubes en 62 países en marzo de 2020 — [The Next Web, 2020-03-11](https://thenextweb.com/plugged/2020/03/11/how-a-budget-ai-camera-is-helping-smaller-football-clubs-record-matches/)
- [T] Prototipo de 2017: «two 4K cameras that produce a full 180-degree panoramic view», IA entrenada con más de un millón de imágenes — [NVIDIA blog, 2017-08-31](https://developer.nvidia.com/blog/ai-technology-automatically-records-soccer-matches)
- [T] Veo Cam 2 (reseña de 2022): cámara de 180°, modo interactivo arrastrable, enlace de descarga del partido, montaje en ~3 minutos — [Keepitonthedeck, 2022-09-21](https://derek-kavanagh-7h2i.squarespace.com/blog/2022/9/21/veo-camera-mark-2-review)

**Pixellot — Air / Air NXT (portátil), Show (fijo HD), Prime (fijo 50-60 fps), DoublePlay (béisbol)**
- [F] Arquitectura de los sistemas fijos: «multiple cameras in a single camera head unit, which itself is powered by a computer tower (video processing unit). The feed from each camera is stitched together to form a panoramic view… From this panoramic view, the relevant algorithm for the sport is applied to create an authentic broadcast feed that follows the action, zooming in/out and panning left/right» — [Pixellot FAQ](https://www.pixellot.tv/faq/)
- [F] «Pixellot Show is a fully AI-Automated set-and-forget HD sports capture and streaming solution for semi-pro and amateur leagues»; «Pixellot Prime… high-resolution video capture at 50-60 FPS with an option to add more camera units» — [Pixellot FAQ](https://www.pixellot.tv/faq/)
- [F] El procesado es local (en la sede): hay «a server in the stadium which the camera is connected to»; «Each of those 40,000 units has three cameras that watch the full field and get a panoramic view»; el resultado «is in the cloud, and they can download it»; retardo de tres segundos para apuestas — [SVG, entrevista a Pixellot, 2025-11-11](https://www.sportsvideo.org/2025/11/11/svg-sit-down-how-pixellots-automated-production-tech-stack-is-evolving-in-the-era-of-ai/)
- [F] Air NXT (presentada en septiembre de 2024): hasta 512 GB, batería de hasta 5 h, micrófonos estéreo, tecnología NVIDIA, «Dual-stream functionality offers both dynamic viewing of TV-style video streams that automatically zoom in on the action or a panoramic, full-field/court option» — [TVBEurope, 2024-09-26](https://tvbeurope.com/live-production/pixellot-debuts-latest-portable-ai-driven-camera)
- [F] Air NXT «records the entire field, tracks the game in real-time, and streams automatically»; se coloca en la línea de medio campo; «Download and break down the game using either Pixellot's video analytics platform or any other platform on the market»; «The only solution that offers a tactical view of all field players and allows you to download the game» — [Pixellot Air](https://www.pixellot.tv/products/pixellot-air/)
- [F] Subida recomendada de 5-10 Mb/s, mínimo 2 Mb/s (la calidad del directo baja, pero «a high-quality copy of the footage is always stored locally»); el cabezal va en el centro, a una altura que cubra las cuatro esquinas; al mover el sistema hay que recalibrar (15-20 min) — [Pixellot FAQ](https://www.pixellot.tv/faq/)
- [F] Pixellot Air (versión desarrollada con el Barça Innovation Hub): cámara ligera que graba más de doce horas sin operador, electricidad ni internet — [FC Barcelona](https://www.fcbarcelona.es/es/noticias/2051982/el-bihub-y-pixellot-desarrollan-un-nuevo-sistema-de-grabacion-automatizado-para-las-barca-academy-y-el-futbol-formativo)
- Parque instalado: [F] «+30,000 Pixellot systems have been sold» en el pie de su web — [Pixellot Air](https://www.pixellot.tv/products/pixellot-air/); «40,000-plus cameras» en la entrevista de 2025 — [SVG, 2025-11-11](https://www.sportsvideo.org/2025/11/11/svg-sit-down-how-pixellots-automated-production-tech-stack-is-evolving-in-the-era-of-ai/). Las cifras no coinciden.
- **Sin verificar** (fichas de minoristas no abiertas): Air NXT con dos sensores de 12 MP (4.056 × 3.040), campo de 200° × 80° y salida «up to 1080p HD».

**Spiideo (Suecia) — cámaras fijas con nube; Micro X1; línea portátil a batería**
- [F] Gama publicada: una línea económica con «2k resolution capture, 180° horizontal panoramic view»; la D-Line con «triple camera sensors and built-in stitching… full panoramic view already at 0.5 m from the sideline», 180° × 90°, IP68; una línea con «4k resolution per camera, 50/60 fps, 100°-170° horizontal field of view» (IP66/IP67); la línea premium con «4k resolution per camera, 70°-170° horizontal with high-end Canon optics» y sensores «8x larger than e.g. GoPro Hero 7»; una línea VAR a 100 fps; Micro X1 «1080p at 50 fps or 720p up to 180 fps» — [Spiideo, sistemas de cámara](https://www.spiideo.com/sports-video-camera-systems/)
- [F] Todo va por nube: «Automated sports video capture with virtual panorama and sport-specific AI AutoFollow… Powered by Spiideo's low-latency cloud»; «Let Spiideo's virtual panorama capture the entire field, while you pan, zoom and switch angles»; servidores en la región del cliente; Spiideo Play admite RTMP, SRT y RTSP — [Spiideo, sistemas de cámara](https://www.spiideo.com/sports-video-camera-systems/)
- [F] Autocast 2.0 (julio de 2024): varios ángulos, CloudStudio Next, Micro X1 y tercera generación de AutoFollow — [SVG, 2024-07-08](https://www.sportsvideo.org/2024/07/08/spiideo-unveils-major-industry-first-product-updates-underscoring-advancement-in-ai-automated-sports-broadcasting/)
- La patente de Spiideo describe que el servidor guarda el vídeo bruto y la calibración, y que el PTZ virtual se calcula **en el cliente** (ver §2) — [US10834305B2](https://patents.google.com/patent/US10834305B2/en)

**Hudl Focus — Focus Flex (portátil), Focus Outdoor/Indoor (fijas), Focus Point**
- [F] Focus Flex: «portable high-definition smart camera», lanzada en EE. UU. en enero de 2023 tras 18 meses de desarrollo y pruebas con más de 90 clubes; se controla con la app Hudl Focus; 4G en EE. UU. — [Hudl, nota de lanzamiento](https://www.hudl.com/blog/hudl-launches-portable-auto-tracking-camera)
- [F] Flex: «uses AI player tracking to automatically pan and zoom to follow the action»; 500 GB («over 100 matches»); batería interna más batería externa; subida por 4G, wifi o punto de acceso; trípode de 7,6 m opcional; precio «tailored», suscripción anual; no publica resolución ni fps — [Hudl Focus Flex FAQ](https://www.hudl.com/en_gb/products/focus/faq/flex)
- [F] Flex: «extra-wide tactical angle ensuring a minimum of 20 players will always be in view»; «HD video including tactical wide angle and practice mode»; calibraciones de fútbol 5 a fútbol 11 — [Hudl Focus Flex](https://www.hudl.com/en_gb/products/focus/flex)
- [F] Focus Outdoor: «Our camera uses four lenses working together to create a single image. It simultaneously sees the whole field while also using player tracking technology to follow the action» — [Hudl blog, 2021-05-07](https://www.hudl.com/blog/training-the-eye-in-the-sky)
- [F] Las Focus fijas se conectan por un único cable Ethernet y graban según el calendario aunque caiga la red — [Hudl Focus, detalles](https://www.hudl.com/en_gb/products/focus/details)
- [C] Veo afirma que Focus Flex es «Pan-tilt-zoom only, no full-field recording… around $1,999 hardware cost… 1080p image quality» — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared). Choca con la descripción de Hudl (multi-lente, ángulo táctico ancho); no hay ficha técnica pública que lo zanje.
- **Sin verificar**: que Focus Outdoor entregue «broadcast and full tactical video angles in 1080p» (visto solo en resumen de búsqueda).

**Trace (precios en dólares; modelo de alquiler)**
- [F] «The Trace camera films the entire field, and AI edits the footage into personalized highlights for every player on both teams»; «PlayerCam… zooms in on and follows your player» — [Trace pricing](https://traceup.com/pricing) y [Trace](https://traceup.com/)
- [F] Hay que introducir colores de camiseta y dorsal; «PlayerFocus AI combines that data with hundreds of other signals to find and follow your player»; trípode de 16 ft; no publica lentes, resolución ni fps — [Trace, how it works](https://traceup.com/how-it-works)
- [F] El equipo se alquila, no se compra; descarga del partido completo solo para suscriptores Pro cuando 7 familias se suscriben a esa cámara, o en planes de club — [Trace pricing](https://traceup.com/pricing)
- [T] «The camera captures a panoramic view of the entire pitch»; trípode de 4 a 16 ft con sacos de arena; con el plan Basic no se descarga nada y solo se ven los cinco últimos partidos — [Core77, 2025-05-21](https://www.core77.com/posts/136929/Good-or-Bad-This-System-Records-Entire-Sporting-Matches-But-Highlights-Just-Your-Child)

**XbotGo (Falcon y Chameleon)**
- [F] Falcon: cámara autónoma de doble lente, «native 4K resolution at 30 frames per second… without any cropping», también 2K y 1080p; chip de IA de 6 TOPS **en el dispositivo**; microSD hasta 1 TB; ~4 h de batería; 20 GB de nube sin suscripción; rosca de 1/4"; los pedidos de Kickstarter se enviaban desde enero de 2026 — [XbotGo Falcon FAQ](https://xbotgo.com/pages/xbotgo-falcon-faq)
- [F] Falcon «Native 4K at 30fps, no cropping or stitching; 1080p at 60fps»; Chameleon es un cabezal para el móvil (720p/1080p/4K «set by your phone»); ninguno lleva 4G/5G; se cita «$799 Falcon» — [XbotGo, Falcon vs Chameleon](https://xbotgo.com/blogs/buying-guide/falcon-vs-chameleon)
- [C] Veo lo clasifica como «Pan-tilt-zoom only, no full-field recording» — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared)

**Pix4Team 2 (Move'N See)**
- [F] Es un robot que mueve (pan/tilt) una cámara o un móvil de terceros de hasta 1,5 kg y controla el zoom de cámaras Sony/Canon compatibles; «follows the action, it does not follow any specific player or the ball»; altura mínima 4,80 m (recomiendan 5-7 m) y de 3 a 20 m del borde del campo — [Move'N See, Pix4Team 2](https://shop.movensee.com/en/30-pix4team-2-team-sports)
- [T] Lleva una cámara integrada de 180° con sensor CMOS de 1/2,5" y 5,1 MP; solo graba en HD (microSD, .mp4); streaming por wifi/Ethernet y RTMP; 2.034 $ en B&H — [Newsshooter, 2024-01-19](https://www.newsshooter.com/2024/01/19/moven-see-pix4team-2-auto-follow-camera/)

**Once Sport Autocam (software)**
- [F] Funciona con cámaras Axis, Dahua, Hikvision, GoPro, DJI, Reolink y otras; montaje fijo con dos cámaras (cada una cubre medio campo) y un mini PC (~1.200 EUR + IVA de hardware) o portátil con dos GoPro/DJI (~600 EUR + IVA), Insta360 Ace 2 o Reolink Duo 3; el recorte «TV-like» (pan y zoom digitales) se procesa en el ordenador del club (Basic hasta 40 h/mes, Pro ilimitado) o en la nube de Once por 5 EUR/partido en 24 h; el vídeo se puede exportar a cualquier plataforma — [Once Sport Autocam](https://once.sport/autocam)

**Zone14 (Austria) — zone14 TWO + REPLAY / LIVE / STATS**
- [F] «180-Grad-4K-Kamera», instalación fija; graba el campo completo los 90 minutos — [zone14](https://zone14.ai/)
- [T, redactado con IA y sin prueba práctica] La zone14 TWO son «zwei 4K-Kameras» — [KI-Syndikat, zone14](https://www.ki-syndikat.de/tools/zone14/)
- [T] En su lanzamiento (fecha del artículo no comprobada) el vídeo se subía por la «zone14 Box» a la nube, donde la IA lo procesa; empresa de Viena — [Trending Topics](https://www.trendingtopics.eu/zone14-wiener-startup-bietet-fussballklubs-ki-gestuetzte-videoanalyse/)

**Staige (web en alemán) — K2X (portátil) y sistemas fijos**
- [F] K2X: cámara móvil lista en 5 minutos, streaming automático en HD, compatible con SIM; «Das System filmt, zoomt und schneidet vollautomatisch»; más de 1.400 sistemas instalados; IA en el borde (edge); participada por adesso SE — [Staige K2X](https://staige.com/staige-sport/staige-k2x-kamera), [Staige clubes](https://staige.com/staige-sport/vereine) y [Staige](https://staige.com/)
- No publica sensores, resolución de captura ni si se descarga la panorámica (páginas abiertas: las tres anteriores).

**AutomaticTV (marca de Grup Mediapro) — ATV One, Ace, Pro, Broadcast**
- [F] «Every AutomaticTV system is composed of one master camera – rig – and one server» (procesado local); la cámara maestra va en el centro del campo; en Multicam se añaden 2 cámaras 4K o 4 FullHD laterales o tras la portería — [AutomaticTV FAQ](https://www.automatic.tv/en/faqs/)
- [F] «High image quality productions with 4K and 6K cameras»; vista táctica «that captures all field players»; integración con Mediacoach, Hudl Sportscode, Nacsport y LongoMatch — [AutomaticTV](https://www.automatic.tv/)
- [F] ATV Pro: «4K cameras at 25/30 fps with an incredible dynamic range» — [ATV Pro](https://www.automatic.tv/en/products/atv-pro/)
- [F] Salida en MP4 guardado en el servidor; protocolos SRT, RTMP, TCP, UDP, MPEG y, según sistema, SDI y NDI; en ATV Pro y Broadcast, «Iso Recording… allows you to record the RAW camera images locally on a server and re-produce the match»; funciona sin internet; «everything you produce belongs to you» — [AutomaticTV FAQ](https://www.automatic.tv/en/faqs/)

**Studio Automated (software «Virtual Director»)**
- [F] IA para cámaras PTZ, panorámicas y cabezales remotos; 15 deportes; servidor local o nube; autocalibración sin calibrar lentes; «In select sports, stream tactical and media views simultaneously»; más de 2.225 sedes — [Studio Automated, Virtual Director](https://studioautomated.com/products/virtual-director/) y [Studio Automated](https://studioautomated.com/)

**Reeplayer (EE. UU.; fabrica en Pittsburgh)**
- [F] Solo disponible en Estados Unidos; plan anual de cámara de 1.188 $; conectividad móvil integrada; «Upload HD games in 5-30 minutes»; mencionan una futura «four-lens solution» y un testimonio elogia el cosido («The stitching is incredible») — [Reeplayer](https://www.reeplayer.com/)
- **Sin verificar** (ayuda en 404): tres sensores 4K, exportación a 1080p.

**BePro — Cerberus (portátil) y Fixed Cam**
- [F] «BEPRO cameras capture the entire pitch in 4K panoramic quality… tactical view and our proprietary 3D video player» — [BePro](https://www.bepro.ai/)
- [F] Cerberus: «uses two internal cameras to capture a wide field of view and stitches them together to create a seamless panoramic video»; por defecto genera la panorámica y a la vez una «AI View» tipo retransmisión que emite por RTMP (Sportscode, Nacsport, SBG) — [BePro, Cerberus OS3](https://www.bepro11.com/news-updates/cerberus-os3-update)
- [F] Cerberus: más de 6 h de batería, lluvia y nieve, subida por wifi, Ethernet o LTE; vídeo disponible «within 30 minutes after the final whistle»; de ~184 grabaciones de partido, ~72 % se subieron completas en una hora — [BePro portable](https://www.bepro.ai/portable-sports-camera)
- [T] El sistema fijo estándar de BePro es de «3-camera panoramic stitching» a 4K — [SoccerTrack v2, arXiv 2508.01802](https://arxiv.org/html/2508.01802v1)

**Otros**
- Mobile Viewpoint BV tiene una patente de 2025 sobre detección de balón y jugadores en vistas virtuales de una panorámica cosida (ver §2) — [US12482114B2](https://patents.google.com/patent/US12482114B2/en)
- MyPlay: cámara fija no tripulada, controlada desde el móvil, con experiencia «similar a una retransmisión televisiva»; acuerdo con Andorra — [Alicante Plaza](https://alicanteplaza.es/myplay-firma-un-acuerdo-con-andorra-para-instalar-sus-camaras-deportivas-con-ia-en-el-pais)

### Inferences
- En Veo, el directo exige seguimiento en la cámara y el vídeo «definitivo» se reprocesa en la nube tras la subida (SteadyView «after every recording», follow-cam «after upload»): es la lectura que concilia sus tres mensajes, pero Veo no lo dice así en ninguna página abierta.
- «4K» significa cosas distintas: en Veo y Spiideo es por sensor (la panorámica cosida es más ancha que 3.840 px, pero no se publica cuánto); en XbotGo es el fotograma final de una cámara que gira; en la follow-cam de Veo, Hudl y Reeplayer el resultado es 1080p o «HD».
- Si la panorámica de dos sensores 4K ronda los 7.000-8.000 px de ancho para 180° (estimación mía, no publicada), un jugador en la banda contraria ocupa muy pocos píxeles; eso explicaría que la follow-cam se quede en 1080p y que un usuario pida 4K en Trustpilot (ver §4).
- Pixellot y AutomaticTV procesan en un servidor en la sede; Spiideo, Veo, Trace, Zone14 y BePro dependen de la nube; XbotGo y Pix4Team hacen todo en el dispositivo.

### Gaps
- Ningún fabricante publica resolución exacta de la panorámica cosida, bitrate ni códec del fichero descargable. Habría que medirlo sobre ficheros reales.
- No pude confirmar si Veo permite **descargar** la panorámica completa (solo consta que se puede ver en modo interactivo/panorámico y que «los vídeos» se descargan).
- Fps de la Veo Cam 3: 30 (soporte, sin verificar) frente a 60 (comparativa de Veo). Sin resolver.
- Hudl Focus Flex: resolución, fps, número de lentes y si se conserva la panorámica; Trace, Staige y Zone14: especificaciones de sensor.
- Veo Cam 2: no encontré ficha técnica abierta; solo la reseña de 2022.

## 2. Qué cuentan del algoritmo de seguimiento y de la cámara virtual; patentes

### Takeaway
Lo público converge en un mismo esquema: cosido → detección de balón y jugadores → estimación de la «zona de acción»
(balón, densidad de jugadores, estado del juego) → recorte suavizado con reglas por deporte. Las patentes abiertas
(Pixellot, Mediapro/AutomaticTV, Veo, Spiideo, Mobile Viewpoint) describen justo eso; la de Veo es la más detallada sobre
el balón (mapa de probabilidad + filtro de partículas + postura de los jugadores). No encontré patentes de Hudl ni de Trace
sobre la cámara.

### Cited Findings

**Patentes abiertas en Google Patents**
- **Pixellot, US10262692B2** «Method and system for automatic television production» (Tamir, Oz, Ridnik; solicitud 2015-03-31, concesión 2019-04-16, activa). Reivindicación 1: varias cámaras cubren el evento entero; se detectan objetos asociados al «playing object»; se derivan posiciones actuales **y estimadas**; la producción selecciona porciones del flujo combinado según «predefined video production rules associated with a type of said sport event»; y, **al perderse el balón**, se estima la región que lo contiene a partir de su posición previa y se amplía el encuadre para incluirla. La reivindicación 2 calibra con esquinas del campo, cruces de líneas, punto de penalti o balizas — [US10262692B2](https://patents.google.com/patent/US10262692B2/en)
- **Pixellot, US10438633B2** «Method and system for low cost television production» (Tamir, Oz; solicitud 2014-05-26, concesión 2019-10-08, activa): el servidor de la sede graba los fotogramas brutos troceados en «pixel tiles»; envía una panorámica a menos de un tercio de la tasa de datos a una sala remota; allí se eligen los encuadres; y el servidor reconstruye los fotogramas en alta resolución con líneas de cosido pasadas de coordenadas de mundo a coordenadas de cámara y alineación geométrica por calibración — [US10438633B2](https://patents.google.com/patent/US10438633B2/en)
- **MediaProduccion SL (AutomaticTV), EP3039854B1** «A method and system for producing a video production» (Sagas, Bennassar, Valverde, Torrent; solicitud 2014-07-22, concesión 2020-04-15, activa): cosido de los fotogramas componentes en una panorámica; selección de una sección por reglas del evento; identificación de objetos con **extracción de fondo**; **mapa de densidad** de objetos para elegir la zona donde hay muchos próximos; una reivindicación dependiente compara la distribución de posiciones con una distribución esperada del evento; el tamaño de la sección (zoom) varía de un fotograma a otro — [EP3039854B1](https://patents.google.com/patent/EP3039854B1/en)
- **Veo Technologies, WO2019141813A1 / US11310418B2 / EP3741101B1** (Taxbøl, Teisbæk, Reinicke; solicitud 2019-01-18; US concedida 2022-04-19; EP concedida 2025-04-09, activas). El área de interés se determina a partir de la posición estimada del balón; el módulo de aprendizaje detecta balón, jugadores, eventos («corner kicks, free kicks or penalties, based on the players' position») y posturas; la red da «a single channel image presenting the probability of the object of interest»; se modelan **al menos 100 partículas** que se proyectan sobre esa imagen y se actualizan con «Newtonian dynamics» (filtro de partículas) — [WO2019141813A1](https://patents.google.com/patent/WO2019141813A1/en) y [US11310418B2](https://patents.google.com/patent/US11310418B2/en)
- La EP de Veo añade que el módulo se entrena con una base de imágenes de tamaño uniforme de un jugador y el balón con posición relativa conocida, y que la postura de cada jugador se empareja con esa base para obtener una probabilidad de dirección del balón que se combina con la distribución del filtro de partículas — [EP3741101B1](https://patents.google.com/patent/EP3741101B1/en)
- **Spiideo, US10834305B2 y US11283983B2** «System and method for providing virtual pan-tilt-zoom, PTZ, video functionality to a plurality of users over a data network» (Linderoth, Ardö, Josephson; solicitud 2017-04-11; concesiones 2020-11-10 y 2022-03-22, activas): cámaras de red envían segmentos brutos; una unidad de calibración guarda posición, orientación e intrínsecos respecto al mundo 3D como metadatos; el servidor almacena y sirve vídeo más metadatos; **cada cliente** elige el PTZ virtual y proyecta en su pantalla; con varias cámaras, el servidor sincroniza los fotogramas — [US10834305B2](https://patents.google.com/patent/US10834305B2/en) y [US11283983B2](https://patents.google.com/patent/US11283983B2/en)
- **Mobile Viewpoint BV, US12482114B2** (Stuurman, Bais; solicitud 2021-06-22, concesión 2025-11-25): cámaras sincronizadas alrededor del campo, panorámica cosida recortando solapes, **vistas de cámara virtual «de-warped» a cuadradas** que se pasan a una red profunda para detectar balón y jugadores — [US12482114B2](https://patents.google.com/patent/US12482114B2/en)

**Lo que cuentan fuera de las patentes**
- [F] Pixellot: el algoritmo debe correr «in less than 10 milliseconds per frame»; lo mínimo es identificar balón y jugadores, distinguirlos de árbitros, entrenadores, banquillo y público, decidir si hacer zoom sobre el balón o mostrar la panorámica, y «predict where the ball will be heading with a lead time of four seconds»; la calibración de los límites del campo sirve para saber qué jugadores están activos — [Pixellot, 2021-03-30](https://www.pixellot.tv/?p=795)
- [F] Pixellot: arquitectura de IA «trained on over five million games… over the last eight years» para «accurate ball-tracking and game-state identification» — [TVBEurope, 2024-09-26](https://tvbeurope.com/live-production/pixellot-debuts-latest-portable-ai-driven-camera); «object-based auto tracking, and the camera also knows how to zoom» — [SVG, 2025-11-11](https://www.sportsvideo.org/2025/11/11/svg-sit-down-how-pixellots-automated-production-tech-stack-is-evolving-in-the-era-of-ai/)
- [F] Pixellot: algoritmos distintos por deporte, con «90,000 hours of live content being produced every month» como muestra de entrenamiento — [Pixellot FAQ](https://www.pixellot.tv/faq/)
- [F] Veo: IA «trained on more than 7 million real-life games over 10 years»; detección automática de dorsales — [Veo vs Trace](https://www.veo.com/en-us/veo-vs-trace); en un gol «we focus on the celebration – not the ball sitting in the net» — [Veo Cam 3](https://www.veo.com/en-us/product/veo-cam-3)
- [F] Spiideo: AutoFollow es específico por deporte y de movimiento «smooth and human-like» — [Spiideo, sistemas de cámara](https://www.spiideo.com/sports-video-camera-systems/); la tercera generación «will identify events and phases of fixtures with high precision» — [SVG, 2024-07-08](https://www.sportsvideo.org/2024/07/08/spiideo-unveils-major-industry-first-product-updates-underscoring-advancement-in-ai-automated-sports-broadcasting/)
- [F] Hudl: algoritmo «trained over hundreds of games and thousands of hours of video—to pinpoint the ball, players, officials, the scoreboard» — [Hudl blog, 2021-05-07](https://www.hudl.com/blog/training-the-eye-in-the-sky); rúbrica interna de calidad: sin tirones ni fotogramas perdidos, sin «ghosting» y sin «blurred images between from one camera lens to the next», poco ruido — [Hudl blog, 2021-05-07](https://www.hudl.com/blog/delivering-the-best-film-quality-possible)
- [F] XbotGo Falcon: «combines target recognition, pose estimation, and trajectory prediction», más zoom automático — [XbotGo Falcon FAQ](https://xbotgo.com/pages/xbotgo-falcon-faq)
- [F] AutomaticTV: estilo de producción configurable («wider or closer zoom levels, faster or slower camera movements, camera switching frequency») siguiendo las pautas de realización de Mediapro; control manual con ATV Director — [AutomaticTV](https://www.automatic.tv/) y [AutomaticTV FAQ](https://www.automatic.tv/en/faqs/)
- [F] BePro Cerberus OS3: «More precise ball tracking, capturing as many players as possible around the ball» y menos zona fuera del campo — [BePro, Cerberus OS3](https://www.bepro11.com/news-updates/cerberus-os3-update)
- [F] Pix4Team 2 sigue «la acción», no el balón ni a un jugador — [Move'N See](https://shop.movensee.com/en/30-pix4team-2-team-sports)

### Inferences
- El estado del juego (saque de esquina, falta, celebración) forma parte del encuadre en Pixellot, Veo y Spiideo: la vista producida no es un simple «seguir el balón», y su estilo (cuánto abre, cuánto anticipa) cambia por fabricante y por versión del modelo.
- La patente de Pixellot reivindica expresamente abrir el plano cuando se pierde el balón; eso explica los tramos de plano muy abierto en su vídeo y es un cambio de escala brusco para un modelo posterior.
- La de Spiideo implica que el vídeo bruto por cámara y la calibración existen en su nube: es el fabricante cuya arquitectura encaja mejor con análisis de terceros sobre vídeo estático calibrado (coherente con SynLoc, §5).
- La patente de Veo usa la postura de los jugadores para inferir dónde está el balón cuando no se ve: la idea es trasladable a un detector de balón propio.

### Gaps
- No pude listar las carteras completas (Justia en 403; la búsqueda por titular de Google Patents es dinámica). Puede haber más patentes de Pixellot (la búsqueda menciona solicitudes de 2019 sobre producción con cámaras «off-the-shelf», sin verificar) y una continuación estadounidense de Veo (US 2023/0092516, sin verificar).
- No encontré patentes de Hudl (Agile Sports Technologies) ni de Trace sobre captura o seguimiento; tampoco de Zone14, Staige, BePro, XbotGo o Reeplayer. No es prueba de que no existan.
- Ningún fabricante publica un blog de ingeniería con arquitectura, métricas o suavizado de la cámara virtual; lo más técnico es el artículo de Pixellot de 2021 y las patentes.

## 3. Analítica automática encima del vídeo y fiabilidad declarada o medida

### Takeaway
La oferta se ha desplazado en 2025-2026 hacia «datos sin GPS» (tracking óptico, radar 2D, distancias, sprints) y eventos
automáticos básicos (goles, tiros, córners, faltas). Las cifras de fiabilidad publicadas son casi inexistentes: solo
Pixellot da un «98 %» sin definir y BePro alega certificación FIFA EPTS; no encontré ninguna medición independiente de los
eventos automáticos de estos productos. Varias «analíticas avanzadas» las etiquetan personas.

### Cited Findings
- [F] Veo Analytics 2 (complemento de pago): «Goals, shots, corners, and free kicks detected automatically»; informe de partido de seis pestañas; «Heat maps, shot maps, pass maps, and 2D radar»; Veo Vision (velocidad y distancia superpuestas); posesión %, pases acertados, minutos y distancia «tracked»; Coach Assist (chat sobre datos) — [Veo Analytics 2](https://www.veo.com/en-us/product/veo-analytics)
- [F] Veo Editor: tiros y goles se recortan solos; «Speedrun automatically skips dead time» — [Veo pricing](https://www.veo.com/en-us/pricing)
- [T] Usuario de Veo (2026-11-03): «minor bugs when identifying the player's jersey»; otro (2026-10-13): «all of the software is for soccer only» — [Trustpilot, Veo](https://ca.trustpilot.com/review/veo.co)
- [F] Pixellot: «The algorithm for the cameras currently has a 98% accuracy rate» (no define la métrica) — [Pixellot, 2021-03-30](https://www.pixellot.tv/?p=795)
- [F] Pixellot: «full video breakdown of games and practices, along with detailed team and player analytics» (plataforma VidSwap), resúmenes automáticos y, en 2025, «full optical… data collection» — [Pixellot Air](https://www.pixellot.tv/products/pixellot-air/) y [SVG, 2025-11-11](https://www.sportsvideo.org/2025/11/11/svg-sit-down-how-pixellots-automated-production-tech-stack-is-evolving-in-the-era-of-ai/)
- [F] Spiideo AutoData Activity: «machine-generated, real-time tracking data» (forma del equipo, compacidad, control del campo, métricas de carrera, mapas y redes de pases, sprints) — [Spiideo, AutoData Activity](https://support.spiideo.com/en/articles/13368675-autodata-soccer-activity)
- [F] Spiideo AutoData Advanced: «well over 1000+ events tagged every game **by our team of analysts**» (etiquetado humano) — [Spiideo, AutoData Advanced](https://support.spiideo.com/en/articles/9306177-autodata-soccer-advanced)
- [F] Zone14 STATS: distancias, sprints, carreras de alta intensidad, mapas de calor y de sprints, compacidad y «dominancia», informes PDF por jugador, todo desde el vídeo de la zone14 TWO — [zone14 STATS](https://zone14.ai/fussball-tracker-ohne-gps/)
- [T, redactado con IA] Zone14 no publica la precisión de esos valores: «Wie genau die Werte sind, nennt zone14… nicht» — [KI-Syndikat, zone14](https://www.ki-syndikat.de/tools/zone14/)
- [F] BePro: datos de eventos y «Positional data & physical metrics without GPS equipment»; «All BEPRO cameras are… (EPTS) certified», con calificación «Well Above or higher across all categories» verificable en la web de FIFA (no lo comprobé allí) — [BePro](https://www.bepro.ai/) y [BePro cameras](https://www.bepro.ai/cameras)
- [T] En SoccerTrack v2 las posiciones y los eventos venían de BePro y se convirtieron a 12 clases «with manual checks and corrections performed to ensure accuracy» — [SoccerTrack v2](https://arxiv.org/html/2508.01802v1)
- [F] Trace: resúmenes personalizados para cada jugador de ambos equipos, «PlayerFocus Analytics» con gráficos de posesión y pases — [Trace, how it works](https://traceup.com/how-it-works)
- [F] AutomaticTV ATV One: «All events of interest are detected automatically - in real time»; clips, resúmenes y marcador por OCR — [ATV One](https://www.automatic.tv/en/products/atv-one/) y [AutomaticTV FAQ](https://www.automatic.tv/en/faqs/)
- [F] Studio Automated: «Automated event detection and highlight creation» — [Studio Automated](https://studioautomated.com/)
- [C] PlayerData (vende GPS): los sistemas solo-cámara «can — and do — confuse players»; «Single camera-derived metrics are unable to achieve third party verified standards» — [PlayerData, guía de compra](https://playerdata.com/en-gb/blog/how-to-choose-a-camera-system-for-your-soccer-club)

### Inferences
- Los eventos automáticos que todos ofrecen (gol, tiro, córner, falta) son los detectables por estado del juego y posición de jugadores; pases, duelos y conducciones se venden como servicio con analistas (Spiideo Advanced, VidSwap) o se derivan de tracking.
- Un producto de terceros compite donde estos sistemas no publican fiabilidad: acciones fiables con umbrales conocidos.
- Que SoccerTrack v2 corrigiera a mano los datos de BePro indica que ni un proveedor con certificación EPTS entrega eventos listos para usar como verdad de referencia.

### Gaps
- Ninguna medición independiente de precisión/recall de eventos automáticos de Veo, Pixellot, Spiideo, Hudl, Zone14 o Trace.
- No abrí el listado de EPTS de FIFA para confirmar la certificación de BePro ni si Spiideo o Veo figuran.
- La analítica de Hudl sobre Focus (Assist, «AI Insights», métricas físicas) solo la vi nombrada en titulares.

## 4. Limitaciones y fallos documentados

### Takeaway
Los propios fabricantes admiten pérdidas breves del balón (más en fútbol, por el tamaño del campo), dependencia de la luz,
del sol de cara y de la altura de montaje, y recalibración al mover la cámara. Las fuentes independientes sólidas son
escasas: las quejas más documentadas de usuarios son de precio, horas de grabación y resolución de la vista producida, no
de seguimiento. No encontré documentación fiable del fallo «sigue un balón de otro campo».

### Cited Findings
- [F] Pixellot: la IA puede hacer que la cámara «lose focus on the ball for a short amount of time before it re-centers itself. This will more often happen at soccer games because the soccer field is large»; el balón es difícil de localizar «in the corner or behind a player, or subject to difficult lighting conditions» — [Pixellot, 2021-03-30](https://www.pixellot.tv/?p=795)
- [F] Pixellot: en portátil hay que calibrar antes de cada partido; al mover un sistema fijo, 15-20 minutos de recalibración; hay que limpiar las lentes de vez en cuando; con solo 2 Mb/s de subida el directo pierde calidad (prefieren 5-10 Mb/s) — [Pixellot, 2021-03-30](https://www.pixellot.tv/?p=795) y [Pixellot FAQ](https://www.pixellot.tv/faq/)
- [F] Veo reconoce que «the AI follow-cam momentarily drifts» y lo compensa con la panorámica conservada — [veo.com/llms.txt](https://www.veo.com/llms.txt)
- [T] Trustpilot de Veo: 4,1/5 con 1.310 reseñas, 34 % de una estrella; quejas fechadas: «Wish the video quality would be 4k» (2026-09-16), «Not worth the price… nickel and dimed» (2026-09-10), «I ran out of recording time for the month» (2026-09-27), errores de dorsal (2026-11-03) — [Trustpilot, Veo](https://ca.trustpilot.com/review/veo.co)
- [T] Reseña de Veo Cam 2 (2022): 39 pruebas con seguimiento «perfect»; resistente al agua y a caídas; sostiene que el vídeo viral de una cámara siguiendo la calva de un árbitro era de un competidor, no de Veo — [Keepitonthedeck, 2022-09-21](https://derek-kavanagh-7h2i.squarespace.com/blog/2022/9/21/veo-camera-mark-2-review)
- [T, redactado con IA y sin prueba práctica: peso bajo] «Bei schlechter Ausleuchtung, Flutlichtspiele… verliert das Tracking an Präzision. Gelegentlich folgt die Kamera dem falschen Spieler oder verliert den Ball kurz» — [KI-Syndikat, Veo](https://www.ki-syndikat.de/tools/veo-kamera/)
- **Sin verificar** (soporte de Veo en 403): colocar el sol o los focos detrás de la cámara, subir el trípode a su altura máxima, a 3-5 m de la banda y en la línea de medio campo, y no mover la cámara tras iniciar la grabación (`https://support.veo.com/hc/en-us/articles/5922516538897`).
- [F] Hudl: el diseño tuvo que contemplar «the angle of the sun and stadium lights… extreme weather and varying stadium sizes»; fallos que vigilan: «ghosting (players looking blurry or "teleporting" due to missed frames) or blurred images between from one camera lens to the next»; un entrenador de la beta dijo al principio que la calidad no superaba a sus cámaras — [Hudl blog, 2021-05-07](https://www.hudl.com/blog/delivering-the-best-film-quality-possible)
- [F] Hudl Focus Flex: la cobertura 4G no está disponible en todas partes — [Hudl Focus Flex FAQ](https://www.hudl.com/en_gb/products/focus/faq/flex)
- [F] Trace: «does not work when placed behind a net or a fence»; necesita sitio para el trípode en la línea central — [Trace pricing](https://traceup.com/pricing)
- [F] BePro Cerberus: antes de OS3, «extended recording could sometimes affect quality due to internal heat build-up» — [BePro, Cerberus OS3](https://www.bepro11.com/news-updates/cerberus-os3-update); ~28 % de las grabaciones tardan más de una hora en subirse — [BePro portable](https://www.bepro.ai/portable-sports-camera)
- [T] Pix4Team 2: sensor «rather small» y grabación solo en HD — [Newsshooter, 2024-01-19](https://www.newsshooter.com/2024/01/19/moven-see-pix4team-2-auto-follow-camera/); el fabricante exige al menos 4,80 m de altura — [Move'N See](https://shop.movensee.com/en/30-pix4team-2-team-sports)
- [F] XbotGo Chameleon: la autonomía real la limita la batería del móvil; ningún modelo lleva 4G/5G — [XbotGo, Falcon vs Chameleon](https://xbotgo.com/blogs/buying-guide/falcon-vs-chameleon)
- [C] Veo sobre XbotGo Falcon: «worst video quality of any system in independent testing. Tracking is highly inconsistent and the camera regularly misses the ball» (no identifica la prueba) — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared)
- [C] Veo sobre los sistemas PTZ: «only record what the tracking algorithm points at» — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared); PlayerData dice lo mismo de las cámaras con seguimiento: lo que pasa sin balón «is oftentimes missed entirely or captured inconsistently» — [PlayerData](https://playerdata.com/en-gb/blog/how-to-choose-a-camera-system-for-your-soccer-club)
- [C] Zone14 sobre Veo: «Das Tracking von Spielern ohne Ball ist jedoch eingeschränkt» — [zone14, comparativa](https://zone14.ai/blog/fussball-kameras/veo-vs-pixellot-vs-zone14-der-grose-videoanalysen-vergleich-im-fusball/)
- [T] Académico: transferir modelos de detección de acciones entre ligas es difícil y «the most impactful feature being broadcasting editing quality», con dominios que van de retransmisión profesional a cámara pilotada por IA — [Cabado et al., CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.html)

### Inferences
- Los fallos que los fabricantes admiten (pérdida breve del balón, esquinas, oclusión, luz) son los mismos que degradan a un detector de balón posterior: la follow-cam hereda el error del seguidor (encuadre sin balón) y la panorámica el de la escala (balón de muy pocos píxeles).
- Como la follow-cam se genera por recorte, un error de encuadre no destruye información si se conserva la panorámica (Veo, Pixellot, Spiideo, BePro, AutomaticTV con ISO); en XbotGo y Pix4Team lo no encuadrado se pierde.
- La altura es la restricción física común: Veo vende trípodes de 5,2 y 7,4 m, Hudl uno de 7,6 m, Trace de 4,9 m (16 ft) y Pix4Team pide 4,8-7 m. Con un punto de vista bajo y lateral aumentan la oclusión y la diferencia de escala entre bandas respecto a la TV.
- La estabilización posterior de Veo (SteadyView) existe porque el viento mueve el mástil: en vídeo sin ese paso (directo, otros fabricantes) cabe esperar vibración.

### Gaps
- No encontré fuente fiable (ni fabricante ni tercero) que documente confusión con balones de otro campo, ni cifras de lluvia o de sol de cara. Los foros (Reddit, grupos de entrenadores) no aparecieron en las búsquedas.
- Sin datos de compresión (bitrate) ni de latencia salvo «3 segundos» de Pixellot para apuestas y la mención genérica de «low-latency» de Spiideo.
- Reseñas independientes de Hudl Focus Flex, Pixellot Air NXT, Trace, Reeplayer, Zone14 y Staige: no encontré ninguna abrible. El artículo de Medium «Switching from Pixellot to Veo» dio 403.

## 5. Publicaciones académicas y datasets de estas empresas o grabados con sus cámaras

### Takeaway
Hay tres piezas verificadas: **Spiideo SoccerNet SynLoc** (Spiideo como coautor; imágenes 4K de instalaciones reales con
atletas sintéticos), **SoccerTrack v2** (10 partidos universitarios en panorámica 4K grabados con BePro, con posiciones y
eventos de BePro corregidos a mano) y **Cabado et al. 2024** (evalúa spotters sobre «AI-piloted cameras» semiprofesionales
y amateur). No encontré datasets públicos grabados con Veo, Pixellot o Hudl.

### Cited Findings
- **Spiideo SoccerNet SynLoc** (Ardö, Nilsson, Cioppa, Magera, Giancola, Liu, Ghanem, Van Droogenbroeck; VISIGRAPP 2025): «dedicated static cameras covering the entire pitch… are now used more and more even in lower tiers»; el dataset son «synthetic athletes rendered on top of images from real world installation of such cameras»; tarea nueva de detección en coordenadas de mundo — [SciTePress](https://www.scitepress.org/Papers/2025/131082)
- El kit `sskit` descarga por defecto «full resolution 4K images» (hay versión «fullhd») y el modelo de cámara es «a standard projective pinhole camera model with radial distortion»; es tarea de reto de SoccerNet — [Spiideo/sskit](https://github.com/Spiideo/sskit)
- Håkan Ardö, primer autor de SynLoc, es coinventor de las patentes de PTZ virtual de Spiideo — [US10834305B2](https://patents.google.com/patent/US10834305B2/en)
- **SoccerTrack v2** (arXiv 2508.01802, agosto de 2025): 10 partidos universitarios completos (~900 min), MP4 en 4K panorámico; «two matches were recorded using a BePro Cerberus system, while the remaining eight matches utilized BePro's standard 3-camera panoramic stitching system»; anotaciones de GSR (coordenadas 2D, dorsal, rol, equipo) y 12 clases de BAS derivadas de datos de BePro con corrección manual — [SoccerTrack v2](https://arxiv.org/html/2508.01802v1)
- **Cabado et al., «Beyond the Premier»** (CVPRW 2024; coautores Cioppa, Giancola, Villa, Guijarro-Berdiñas, Padrón, Ghanem, Van Droogenbroeck; DOI 10.1109/CVPRW63382.2024.00343): analiza la transferencia de spotters «across leagues from amateur to professional and broadcast quality from AI-piloted camera to professional broadcast editing» — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.html) y [web del autor](https://brunocabado.com/research/2/)
- **Sin verificar** (PDF no legible en esta sesión; visto en resumen de búsqueda): los dominios serían SoccerNet (retransmisión), Superliga suiza (cámara principal sin editar), BGL Ligue de Luxemburgo (semiprofesional, cámara pilotada por IA) y liga amateur de Sevilla (cámara pilotada por IA). No pude comprobar marcas de cámara ni las cifras de caída de mAP.
- [F] «Seit 2019 setzt RTL Luxemburg auf die Kamerasysteme von STAIGE» — [Staige K2X](https://staige.com/staige-sport/staige-k2x-kamera)
- [F] Pixellot y el Barça Innovation Hub tienen un acuerdo de codesarrollo (el club como banco de pruebas), sin publicación académica localizada — [FC Barcelona](https://www.fcbarcelona.es/es/noticias/1712342/el-barca-innovation-hub-y-pixellot-desarrollaran-sistemas-de-grabacion-automatizada-de-entrenamientos-y-partidos/amp)

### Inferences
- Es plausible que el vídeo luxemburgués de Cabado et al. proceda de cámaras Staige (RTL Luxemburgo las usa desde 2019), pero es una conjetura mía: el artículo no lo dice en lo que pude leer.
- Para entrenar o evaluar sobre el dominio real, lo único público con panorámica 4K y etiquetas de acción es SoccerTrack v2 (BePro); SynLoc da geometría y localización, no acciones ni balón.
- La línea de SoccerNet con Spiideo (cámara estática calibrada, coordenadas de mundo) indica hacia dónde va la investigación: trabajar sobre la panorámica, no sobre la vista producida.

### Gaps
- Ningún dataset o artículo localizado que use vídeo de Veo, Pixellot o Hudl Focus (apareció un proyecto «VEO_Match» en Roboflow Universe, sin descripción ni verificación).
- No pude leer el cuerpo de Cabado et al. (marcas de cámara, resolución, número de partidos, mAP por dominio). Conviene abrir el PDF dentro del contenedor.
- No abrí el informe de resultados de SoccerNet 2026 para la tarea SynLoc; el dato de «cámaras estáticas 4K; atletas lejanos difíciles» queda sin verificar.

## 6. Presencia en España y en el fútbol base; precios y suscripción

### Takeaway
En España el despliegue federativo más visible en 2025-2026 es **Hudl Focus Flex con FootballClub** (PlayRFAF en Andalucía,
más de 3.000 clubes de base según Hudl), seguido de **Veo** (colaborador oficial de la FFCV desde 2024) y **Pixellot**
(Barça, socio ATM Media en Barcelona). AutomaticTV es de Mediapro y en 2019 generaba la cámara táctica de LaLiga. No hay
acuerdo de ámbito nacional de la RFEF con ningún fabricante en lo encontrado. Los precios de entrada van de ~800 $ sin
cuota (XbotGo) a ~1.300 € + ~50 €/mes (Veo) y ~130 €/mes con cámara incluida (Zone14); Hudl, Spiideo, Pixellot,
AutomaticTV y BePro no publican precios.

### Cited Findings

**Acuerdos en España**
- Hudl + FootballClub + RFAF (2025-07-08): crean PlayRFAF; «Hudl Focus Flex cameras will enable automatic filming of matches without the need for camera operators»; FootballClub produce y distribuye — [Hudl blog, 2025-07-08](https://www.hudl.com/blog/hudl-footballclub-partnership)
- Hudl + FootballClub (2026-07-30): renovación plurianual; «3,000+ grassroots clubs across Spain»; en PlayRFAF las retransmisiones semanales pasaron de ~700 en enero de 2026 a más de 2.500 en abril-mayo; trabajan para extender el modelo a otras federaciones territoriales; más de 75.000 jugadores sub-14 a sub-23 seguidos en Wyscout — [Hudl blog, 2026-07-30](https://www.hudl.com/blog/hudl-footballclub-partnership-announcement)
- Veo + FFCV (mayo de 2024): Veo «se convierte desde hoy en nuevo colaborador oficial» de la federación valenciana — [FFCV, 2024-05](https://ffcv.es/wp/blog/2024/05/veo-y-la-ffcv-se-alian-para-impulsar-el-video-en-los-clubes-de-la-comunitat-valenciana/); en marzo de 2025 sortearon un lote de cámara, trípode y suscripción — [FFCV, etiqueta Veo](https://ffcv.es/wp/blog/tag/veo)
- **Sin verificar**: descuento de 200 € en la Veo Cam 3 para clubes valencianos y uso por la Selecció Valenciana (resumen de búsqueda); acuerdo de dos años Veo-FCF con sorteo de una cámara y descuento de 200 € (PDF de la FCF ilegible: `https://files.fcf.cat/pdfs/noticies/1031153.pdf`); acuerdo FFCV con Fly-Fut para grabar con drones.
- Pixellot + ATM Media (Barcelona, 2022-05-04): ATM Media pasa a ser su «key partner in Spain»; adapta la tecnología con algoritmos propios a 16 disciplinas; la nota dice que la tecnología ya la usan FC Barcelona, Real Madrid, Real Sociedad y Real Betis — [SVG, 2022-05-04](https://www.sportsvideo.org/2022/05/04/atm-media-pixellot-ink-deal-for-automated-production-broadcasting-in-spain/)
- Pixellot + FC Barcelona: cámaras fijas en la Ciutat Esportiva y el Johan Cruyff y acuerdo de innovación con el BIHUB — [FC Barcelona](https://www.fcbarcelona.es/es/noticias/1712342/el-barca-innovation-hub-y-pixellot-desarrollaran-sistemas-de-grabacion-automatizada-de-entrenamientos-y-partidos/amp); la Pixellot Air se pensó para las Barça Academy y el fútbol formativo — [FC Barcelona](https://www.fcbarcelona.es/es/noticias/2051982/el-bihub-y-pixellot-desarrollan-un-nuevo-sistema-de-grabacion-automatizado-para-las-barca-academy-y-el-futbol-formativo)
- Pixellot Air cubrió la Tximist Cup femenina de cantera (Hondarribia y San Sebastián, 13-16 de junio de 2024) — [TVBEurope, 2024-05-23](https://www.tvbeurope.com/live-production/spanish-womens-football-competition-scores-with-ai-driven-cameras)
- AutomaticTV es marca de Grup Mediapro — [AutomaticTV](https://www.automatic.tv/); en LaLiga se usaba para entregar la cámara táctica a los cuerpos técnicos — [SVG Europe, 2019-08-12](https://www.svgeurope.org/blog/headlines/countdown-to-kick-off-how-mediapro-is-preparing-for-the-new-laliga-season/)
- Spiideo: UB Conquense (Segunda Federación) firma por tres años (2026-08-04) — [SVG Europe, 2026-08-04](https://www.svgeurope.org/blog/news-roundup/ub-conquense-brings-spiideos-performance-and-broadcasting-ecosystem-to-castile-la-mancha/); la Federación Española de Baloncesto lo eligió para la producción automática de sus campeonatos — [SVG, 2025-07-07](https://www.sportsvideo.org/2025/07/07/spanish-basketball-federation-chooses-spiideo-as-provider-of-automatic-productions/)
- Staige cita como cliente al responsable de análisis y big data de la academia de Osasuna — [Staige clubes](https://staige.com/staige-sport/vereine)
- MyPlay (cámaras fijas con IA) firmó con Andorra para sus instalaciones deportivas; que tenga unos 40 clubes en España queda **sin verificar** (resumen de búsqueda) — [Alicante Plaza](https://alicanteplaza.es/myplay-firma-un-acuerdo-con-andorra-para-instalar-sus-camaras-deportivas-con-ia-en-el-pais)

**Precios y modelo**
- Veo (EE. UU.): «Veo Cam 3 from $67/month» o «One-time camera purchase $1,299»; «Veo Go from $40/month» — [Veo vs Trace](https://www.veo.com/en-us/veo-vs-trace). Facturación mensual, a 6 meses, anual o a 2 años; Veo Live y Veo Analytics 2 son complementos — [Veo pricing](https://www.veo.com/en-us/pricing)
- [C] Veo en euros (octubre de 2026, según zone14): cámara desde 1.299 € (5G: 1.599 €), trípode y bolsa aparte, suscripción desde 49 €/mes para un equipo — [zone14, comparativa](https://zone14.ai/blog/fussball-kameras/veo-vs-pixellot-vs-zone14-der-grose-videoanalysen-vergleich-im-fusball/)
- [C] Pixellot, según zone14: cámaras «von 800 € bis zu über 5.000 €» más licencia anual — [zone14, comparativa](https://zone14.ai/blog/fussball-kameras/veo-vs-pixellot-vs-zone14-der-grose-videoanalysen-vergleich-im-fusball/)
- Zone14: paquete con cámara 130 €/mes (1.560 € facturados al año), patrocinio de hasta 1.400 € que cubre parte del coste, software solo 20 €/mes (en la página aparece además la cifra 1.700 €, cuyo contexto no comprobé); según su comparativa, la cuota vale para todos los equipos del club — [zone14, comprar cámara](https://zone14.ai/fussball-kamera-kaufen/)
- Hudl Focus Flex: precio a medida, suscripción anual — [Hudl Focus Flex FAQ](https://www.hudl.com/en_gb/products/focus/faq/flex); [C] Veo lo sitúa en «around $1,999» de hardware más suscripción opaca — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared)
- Trace: 180 $/año (Basic) o 300 $/año (Pro) con alquiler de cámara gratis, o mensual más 99 $ de alquiler; paga cada familia — [Trace pricing](https://traceup.com/pricing)
- XbotGo Falcon: 799 $, sin suscripción — [XbotGo, Falcon vs Chameleon](https://xbotgo.com/blogs/buying-guide/falcon-vs-chameleon)
- Pix4Team 2: 2.034 $ (enero de 2024) — [Newsshooter](https://www.newsshooter.com/2024/01/19/moven-see-pix4team-2-auto-follow-camera/)
- Once Autocam: hardware estimado de 600-1.200 EUR + IVA y 5 EUR por partido si se procesa en su nube — [Once Sport Autocam](https://once.sport/autocam)
- Reeplayer: 1.188 $/año, solo EE. UU. — [Reeplayer](https://www.reeplayer.com/)
- Veo dice tener más de 40.000 clubes en 100 países — [Veo, comparativa 2026](https://www.veo.com/en-us/article/soccer-camera-systems-compared)

### Inferences
- En el fútbol base español, el vídeo que más va a crecer a corto plazo es el de **Hudl Focus Flex vía FootballClub** (retransmisiones federativas) y el de **Veo** comprado por los clubes; en Cataluña, además, Pixellot (Barça, ATM Media) y AutomaticTV (Mediapro).
- El vídeo de PlayRFAF es en origen una retransmisión (vista producida, con rótulos y marcador), no un fichero de análisis: es el peor caso para modelos posteriores, salvo que el club baje la grabación de su cuenta Hudl.
- Trace y Reeplayer no son relevantes en España (modelo de alquiler por familias en EE. UU. y venta solo en EE. UU.).

### Gaps
- Sin precios públicos de Hudl, Spiideo, Pixellot, AutomaticTV, BePro ni Staige; sin precios oficiales de Veo en euros (la página los carga dinámicamente).
- No hallé acuerdos de la RFEF ni de la FCF verificables con fabricantes de cámaras, ni el proveedor de cámaras de Footters.
- Presencia de BePro, Zone14, Once y XbotGo en clubes españoles: sin datos.
- No verifiqué el acuerdo AutomaticTV-Valencia CF (2017) ni la cifra de 842 partidos producidos (páginas en 404).

## 7. Implicación práctica: qué vídeo recibiría un producto de análisis de terceros y con qué obstáculos

### Takeaway
Lo normal es recibir un **MP4 de la vista producida a 1080p** (25-30 fps), con encuadre decidido por la IA del fabricante.
La panorámica completa existe en casi todos los sistemas de recorte, pero solo algunos dicen claramente que se puede
bajar o emitir (Pixellot «tactical view», BePro panorámica por defecto, AutomaticTV vista táctica e ISO, Spiideo en su
nube). El obstáculo principal no es técnico sino de acceso: el vídeo vive en la nube del fabricante bajo la cuenta del
club, sin API pública documentada, y con límites por plan.

### Cited Findings
- Veo: descarga desde el Veo Editor y clips MP4; la follow-cam es 1080p; la panorámica se ve en modo interactivo — [Veo Cam 3](https://www.veo.com/en-us/product/veo-cam-3) y [Veo Editor](https://www.veo.com/product/veo-editor); límites de horas de grabación por plan (10/20/200 h al mes) — [Veo pricing](https://www.veo.com/en-us/pricing)
- Pixellot: doble flujo (producido y panorámico) y descarga para usar «any other platform on the market» — [TVBEurope, 2024-09-26](https://tvbeurope.com/live-production/pixellot-debuts-latest-portable-ai-driven-camera) y [Pixellot Air](https://www.pixellot.tv/products/pixellot-air/)
- Spiideo: el vídeo y la panorámica virtual están en su nube; admite RTMP, SRT y RTSP — [Spiideo, sistemas de cámara](https://www.spiideo.com/sports-video-camera-systems/); menciona una «event API» — [SVG, 2024-07-08](https://www.sportsvideo.org/2024/07/08/spiideo-unveils-major-industry-first-product-updates-underscoring-advancement-in-ai-automated-sports-broadcasting/)
- Hudl Focus Flex: sube solo a la biblioteca Hudl del club; la FAQ no describe la descarga — [Hudl Focus Flex FAQ](https://www.hudl.com/en_gb/products/focus/faq/flex)
- BePro: panorámica por defecto más «AI View»; RTMP hacia Sportscode o Nacsport — [BePro, Cerberus OS3](https://www.bepro11.com/news-updates/cerberus-os3-update) y [BePro cameras](https://www.bepro.ai/cameras)
- AutomaticTV: MP4 en el servidor del club, varias salidas simultáneas (programa, cámaras laterales por SRT «to a third party for tactical analysis»), grabación ISO de las cámaras en bruto en Pro/Broadcast, y el contenido «belongs to you» — [AutomaticTV FAQ](https://www.automatic.tv/en/faqs/)
- Trace: partido completo descargable solo en Pro con 7 familias suscritas o en plan de club — [Trace pricing](https://traceup.com/pricing)
- Once Autocam: el club tiene el bruto de sus propias cámaras y el vídeo procesado, exportable a cualquier plataforma — [Once Sport Autocam](https://once.sport/autocam)
- XbotGo Falcon: fichero local en microSD, 4K30 o 1080p60, sin nube obligatoria — [XbotGo Falcon FAQ](https://xbotgo.com/pages/xbotgo-falcon-faq)
- Pix4Team: el vídeo es el de la cámara de terceros o HD en microSD; un cliente destaca «we own our footage» — [Move'N See](https://shop.movensee.com/en/30-pix4team-2-team-sports)
- Studio Automated: «Open API» y vistas táctica y de medios simultáneas en algunos deportes — [Studio Automated, Virtual Director](https://studioautomated.com/products/virtual-director/)

### Inferences
Tabla de síntesis (deducida de los hallazgos citados en §1-§7; «?» = no documentado):

| Sistema | Lo que llegará por defecto | ¿Panorámica completa accesible? | Obstáculo principal |
|---|---|---|---|
| Veo Cam 3 / Go | MP4 follow-cam 1080p, encuadre IA, estabilizado | Se ve en el Editor; descarga «?» | Nube de Veo, cuenta del club, horas por plan, sin API pública localizada |
| Pixellot Air NXT / Show | MP4 «TV-style» HD | Sí: flujo panorámico/táctico y descarga declarada | Depende del plan y del socio local (ATM Media); recalibración |
| Spiideo | Recorte AutoFollow desde su nube | Sí en su plataforma (PTZ virtual); exportación «?» | Todo en nube Spiideo; precio institucional |
| Hudl Focus Flex | Vídeo HD con seguimiento en Hudl; en PlayRFAF, retransmisión con grafismo | «?» (ángulo táctico ancho declarado) | Ecosistema cerrado de Hudl; sin ficha técnica |
| BePro Cerberus / fija | Panorámica 4K + «AI View» | Sí, es la salida por defecto | Nube BePro Space |
| AutomaticTV | MP4 de programa; vista táctica | Sí: vista táctica e ISO en bruto (Pro/Broadcast) | Sistema caro de sede fija; depende de la configuración |
| Zone14 | Vídeo 180° 4K en REPLAY | «?» | Nube; sin presencia conocida en España |
| Trace | Clips por jugador; partido en streaming | Descarga muy restringida | Alquiler por familias en EE. UU. |
| XbotGo Falcon | MP4 4K30/1080p60 de cámara que gira | No existe (no graba campo completo) | Lo no encuadrado se pierde; seguimiento irregular según un competidor |
| Pix4Team 2 | Vídeo de la cámara montada (óptica y zoom reales) | No existe | Igual que arriba; requiere 5-7 m de altura |
| Once Autocam | Recorte «TV-like» + brutos de las cámaras del club | Sí: el club tiene los brutos | Calidad dependiente de cámaras de consumo (GoPro, Reolink) |

- Para el producto conviene pedir a los clubes, por este orden: (1) la panorámica o vista táctica si el sistema la exporta (Pixellot, BePro, AutomaticTV, Once); (2) la follow-cam a máxima calidad; y evitar la retransmisión con grafismo.
- Cada fabricante tiene su «firma» de vista producida (nivel de zoom, anticipación, apertura al perder el balón, estabilización): conviene tratar cada sistema como un dominio distinto al evaluar y, si se puede, identificar el sistema de origen.
- Con panorámica, la entrada deja de parecerse a la TV (campo completo, jugadores pequeños, distorsión de gran angular): lo razonable es generar una cámara virtual propia sobre ella, como hacen los fabricantes.
- Obstáculos legales previsibles (no verificados en términos de servicio): el titular de la cuenta es el club, hay menores en las imágenes (RGPD) y los datos pueden salir de la UE; AutomaticTV es el único que afirma por escrito que el contenido es del cliente.

### Gaps
- No abrí los términos de servicio ni los contratos de encargo de tratamiento de ningún fabricante: propiedad del vídeo, licencia de uso para entrenar su IA y cesión a terceros quedan sin verificar.
- No encontré documentación de API pública de vídeo de Veo, Hudl, Pixellot ni BePro; Spiideo y Studio Automated mencionan API sin detalle.
- Resolución, fps reales, bitrate y GOP de los ficheros descargados de cada sistema: hay que medirlos con ficheros de clubes.
- Si la descarga de Veo ofrece la panorámica (y a qué resolución) es la incógnita con más impacto y debería preguntarse a un club usuario.
