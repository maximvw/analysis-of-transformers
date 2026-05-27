# Анализ трансформеров в задачах пошаговых рассуждений

## Что смотреть в первую очередь

1. Папку `notebooks/` - здесь находятся основные ноутбуки с постановкой экспериментов, обучением моделей и сравнением результатов.
2. Ноутбуки по кратчайшему пути:
   `notebooks/shortest-path-dijkstra-train.ipynb`,
   `notebooks/shortest-path-dijkstra-train-2-4-6.ipynb`,
   `notebooks/shortest-path-dijkstra-train-10-14.ipynb`,
   `notebooks/shortest-path-dijkstra-compare-depth-scaling.ipynb`,
   `notebooks/shortest-path-dijkstra-compare-lambda-sweep.ipynb`,
   `notebooks/shortest-path-dijkstra-compare-aux-delta.ipynb`,
   `notebooks/shortest-path-dijkstra-compare-depth-bridge.ipynb`.
3. Ноутбуки по отслеживанию переменных:
   `notebooks/variables_tracking_train.ipynb`,
   `notebooks/variables-tracking-train-fixed-length.ipynb`.
4. Ноутбуки по задачам на графах:
   `notebooks/graph_train.ipynb`,
   `notebooks/graph-train-fixed-length.ipynb`.
5. После ноутбуков имеет смысл смотреть исходный код конкретных задач в папках `shortest_path_dijkstra/`, `variables_tracking/`, `graph_connectivity/`, `topological_graph_sorting/` и `string_edit_sequence/`.

Если нужен самый короткий маршрут по репозиторию, то достаточно начать с `notebooks/shortest-path-dijkstra-train.ipynb`, затем посмотреть ноутбуки сравнения по глубине и параметрам, а после этого перейти к `variables_tracking`.

## Структура репозитория

- `notebooks/` - основные ноутбуки с экспериментами и сравнением результатов.
- `shortest_path_dijkstra/` - обучение и оценка моделей на задаче поиска кратчайшего пути.
- `variables_tracking/` - эксперименты по отслеживанию состояния переменных в пошаговых вычислениях.
- `graph_connectivity/` - задача связности графа.
- `topological_graph_sorting/` - задача топологической сортировки.
- `string_edit_sequence/` - задача редактирования строк как последовательности шагов.
- `llm_auxiliary/` - код для вспомогательных экспериментов с дополнительными сигналами обучения.
- `Internalize_CoT_Step_by_Step/` - код и данные для экспериментов, связанных с ICoT и обучением на пошаговых рассуждениях.
- `icot/` - отдельные скрипты анализа и воспроизведения экспериментов по implicit chain-of-thought.
