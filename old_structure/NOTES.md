models.py - model klasyfikacji i różne konfiguracje modeli SAE

config.py - konfiguracja

Zamiast używać gotowego datasetu Colored MNIST stworzyłem swój wrapper, który na zwykły MNIST nakłada kolory w zależności od klasy. Przy inicjalizacji podaje się argument jaki kolor datasetu ma być pokolorowany. Z eksperymentów wynikło, że dla 95% pokolorwanego datasetu model skupia się mocno na kolorach, ale w miarę radzi sobie też dla liczb białych. Widać natomiast że model przede wszystkim skupia się na kolorze bo jak odwórci się kolory (że np. jak wcześniej 1 było zielone a 8 czerwone to będzie na odwrót) to skuteczność modelu spada prawie do poziomu przypadkowego zgadywania klasy.

Tu przykładowe wyniki:
```bash
Performance on clean validation set (only white):
Clean Val Loss: 0.0146, Accuracy: 0.8698, Precision: 0.8698, Recall: 0.8698, F1 Score: 0.8696, AUPRC: 0.9161, AUROC: 0.9794

Performance on fully colored validation set:
Fully Colored Val Loss: 0.0001, Accuracy: 0.9988, Precision: 0.9988, Recall: 0.9988, F1 Score: 0.9987, AUPRC: 0.9998, AUROC: 0.9999

Performance on validation set with colors flipped:
Colors flipped Val Loss: 0.1509, Accuracy: 0.3181, Precision: 0.3181, Recall: 0.3181, F1 Score: 0.2966, AUPRC: 0.3592, AUROC: 0.7136
```

Żeby dostać takie wyniki trzeba najpierw skonfigurować jaki procent datasetu ma być pokolorowany w config.py, potem uruchomić prepare_dataset.ipynb, który zapisze przygotowane dane w folderze "data". Potem należy uruchomić cls_train.ipynb.

Trening SAE robi się w sae_train.ipynb. Najważniejsza konfiguracja jest poprzez stałe:

- SAE_FEATURES_LAYER - indeks warstwy modelu klasyfikacji, której aktywacje będą używane do trenignu SAE

- SAE_INPUT_FEATURES - ile neuronów ma mieć na wejściu SAE (równe liczbie neuronów w analizowanej warstwie modelu do klasyfikacji)

- SAE_EXPANSION_FACTOR - ilu krotnie warstwa ukryta SAE ma być większa od wejścia

- SAE_ALPHA - używa modelu SAE, który wymusza rzadkość wektora ukrytego poprzez dodanie do funkcji straty L1. Jest to waga używana do kontrolowania jaki wpływ ma mieć rzadkość wektora ukrytego w porównaniu z błędem rekonstrukcji. Jest używane tylko gdy SAE_TOPK=0

- SAE_TOPK - liczba neuronów z maksymalną aktywacją, które nie będą wyzerowane. Stosuje TopKSAE

- MATRYOSHKA_SAE - Stosuje MatryoshkaBatchTopkSAE

- MATRYOSHKA_EXPANSION_FACTORS - definiuje wielkość poszczególnych "podwektorów" ukrytych w SAE

Zaimplementowany jest również BatchTopkSAE, który wybiera TopK*batchsize neuronów spośród batcha. W teorii powinno wmuszać, żeby średnio było aktywnych TopK neuronów, ale możliwe jest zastosowanie w konkretnych przypadkach mniej lub więcej neuronów warstwy ukrytej do rekonstrukcji wejścia przez SAE.


# Wyniki
Chyba najsensowniejsze wyniki dostałem dla commit: c76efa2. Modyfikowałem tam stan ukryty poprzez: encoding SAE -> ustawienie wartości neuronów, które zidentyfikowałem za reagujące na kolor na 0 -> decoding SAE. To podejście może nie być optymalne, ponieważ bardzo zależy tego w jaki sposób SAE robi rekonstrukcję. Gemini i Claude mi zasugerowali że przeważnie z SAE bierze się "wadliwy" kierunek (czyli wagi dekodera) i odejmuje się ten kierunek od oryginalnego wektora (tego który wchodzi na wejście SAE) i wtedy to jak dobrze SAE rekonstruuje wejście nie jest istotne.

Tutaj output:
```bash
PRZED MODYFIKACJA

Performance on clean validation set:
Clean Val Loss: 0.015241414661208789, Accuracy: tensor([0.9705, 0.9199, 0.9027, 0.7382, 0.8134, 0.8192, 0.7880, 0.8707, 0.8333,
        0.9168]), Precision: tensor([0.9705, 0.9199, 0.9027, 0.7382, 0.8134, 0.8192, 0.7880, 0.8707, 0.8333,
        0.9168]), Recall: tensor([0.9705, 0.9199, 0.9027, 0.7382, 0.8134, 0.8192, 0.7880, 0.8707, 0.8333,
        0.9168]), F1 Score: tensor([0.9013, 0.9484, 0.8591, 0.8272, 0.8308, 0.7939, 0.8623, 0.8738, 0.8340,
        0.8338]), AUPRC: tensor([0.9622, 0.9808, 0.9220, 0.8691, 0.8938, 0.8188, 0.9333, 0.9535, 0.9011,
        0.8876]), AUROC: tensor([0.9947, 0.9939, 0.9792, 0.9444, 0.9721, 0.9556, 0.9831, 0.9880, 0.9765,
        0.9785])

Performance on fully colored validation set:
Fully Colored Val Loss: 0.00024403737605704616, Accuracy: tensor([1.0000, 1.0000, 1.0000, 0.9984, 1.0000, 1.0000, 1.0000, 1.0000, 0.9991,
        0.9992]), Precision: tensor([1.0000, 1.0000, 1.0000, 0.9984, 1.0000, 1.0000, 1.0000, 1.0000, 0.9991,
        0.9992]), Recall: tensor([1.0000, 1.0000, 1.0000, 0.9984, 1.0000, 1.0000, 1.0000, 1.0000, 0.9991,
        0.9992]), F1 Score: tensor([1.0000, 1.0000, 1.0000, 0.9988, 1.0000, 0.9995, 0.9992, 1.0000, 0.9996,
        0.9996]), AUPRC: tensor([1.0000, 1.0000, 1.0000, 0.9996, 1.0000, 1.0000, 1.0000, 1.0000, 0.9994,
        0.9995]), AUROC: tensor([1.0000, 1.0000, 1.0000, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 0.9997,
        0.9999])

Performance on validation set with colors flipped:
Colors flipped Val Loss: 0.15280315073331197, Accuracy: tensor([0.6194, 0.0475, 0.7727, 0.9763, 0.0000, 0.0000, 0.8716, 0.0016, 0.0145,
        0.0000]), Precision: tensor([0.6194, 0.0475, 0.7727, 0.9763, 0.0000, 0.0000, 0.8716, 0.0016, 0.0145,
        0.0000]), Recall: tensor([0.6194, 0.0475, 0.7727, 0.9763, 0.0000, 0.0000, 0.8716, 0.0016, 0.0145,
        0.0000]), F1 Score: tensor([0.6355, 0.0498, 0.5642, 0.9326, 0.0000, 0.0000, 0.9194, 0.0022, 0.0136,
        0.0000]), AUPRC: tensor([0.3126, 0.1576, 0.3761, 0.9913, 0.0504, 0.0486, 0.9820, 0.1354, 0.2072,
        0.0647]), AUROC: tensor([0.8030, 0.6766, 0.9062, 0.9989, 0.0022, 0.0627, 0.9980, 0.6661, 0.8165,
        0.2537])


===========
PO MODYFIKACJI

Performance on clean validation set:
Clean Val Loss: 0.040313775435090064, Accuracy: tensor([0.8641, 0.7678, 0.6779, 0.5669, 0.3339, 0.8201, 0.7035, 0.0040, 0.8427,
        0.9782]), Precision: tensor([0.8641, 0.7678, 0.6779, 0.5669, 0.3339, 0.8201, 0.7035, 0.0040, 0.8427,
        0.9782]), Recall: tensor([0.8641, 0.7678, 0.6779, 0.5669, 0.3339, 0.8201, 0.7035, 0.0040, 0.8427,
        0.9782]), F1 Score: tensor([0.8397, 0.8240, 0.7240, 0.6535, 0.4921, 0.6712, 0.7451, 0.0079, 0.7706,
        0.5292]), AUPRC: tensor([0.7251, 0.9248, 0.7879, 0.6834, 0.7171, 0.6798, 0.7085, 0.8250, 0.8689,
        0.6923]), AUROC: tensor([0.9553, 0.9748, 0.9554, 0.8811, 0.9196, 0.9350, 0.9350, 0.9406, 0.9727,
        0.9605])

Performance on fully colored validation set:
Fully Colored Val Loss: 0.04802806063493093, Accuracy: tensor([9.5949e-01, 0.0000e+00, 4.8154e-01, 1.0000e+00, 8.5616e-04, 8.7362e-01,
        8.0236e-01, 9.5770e-03, 9.9915e-01, 9.9832e-01]), Precision: tensor([9.5949e-01, 0.0000e+00, 4.8154e-01, 1.0000e+00, 8.5616e-04, 8.7362e-01,
        8.0236e-01, 9.5770e-03, 9.9915e-01, 9.9832e-01]), Recall: tensor([9.5949e-01, 0.0000e+00, 4.8154e-01, 1.0000e+00, 8.5616e-04, 8.7362e-01,
        8.0236e-01, 9.5770e-03, 9.9915e-01, 9.9832e-01]), F1 Score: tensor([0.9776, 0.0000, 0.6093, 0.7799, 0.0016, 0.9006, 0.8625, 0.0190, 0.7252,
        0.4975]), AUPRC: tensor([0.9949, 0.0914, 0.7714, 0.9994, 0.3482, 0.9398, 0.9773, 0.7100, 0.9980,
        0.9648]), AUROC: tensor([0.9983, 0.4434, 0.9800, 0.9999, 0.8775, 0.9845, 0.9969, 0.9497, 0.9997,
        0.9942])

Performance on validation set with colors flipped:
Colors flipped Val Loss: 0.08413383502761523, Accuracy: tensor([0.5004, 0.0185, 0.5185, 1.0000, 0.0154, 0.0028, 0.6132, 0.0000, 0.9983,
        0.1319]), Precision: tensor([0.5004, 0.0185, 0.5185, 1.0000, 0.0154, 0.0028, 0.6132, 0.0000, 0.9983,
        0.1319]), Recall: tensor([0.5004, 0.0185, 0.5185, 1.0000, 0.0154, 0.0028, 0.6132, 0.0000, 0.9983,
        0.1319]), F1 Score: tensor([0.5749, 0.0228, 0.6574, 0.6008, 0.0279, 0.0040, 0.6753, 0.0000, 0.6659,
        0.0775]), AUPRC: tensor([0.4883, 0.1047, 0.8221, 0.9994, 0.0641, 0.0950, 0.7880, 0.2556, 0.9428,
        0.1142]), AUROC: tensor([0.8709, 0.4975, 0.9488, 0.9999, 0.2982, 0.5810, 0.9676, 0.8388, 0.9953,
        0.6187])
```

Przede wszystkim trzeba patrzeć na performance dla colors flipped dataset. Widać że dla niektórych klas accuracy spadło, ale dla klasy "8" wzrosła z 0.0145 do 0.9983 (sumaryczny wzrost dla wszystkich klas to było jakoś z 0.31 do 0.46)


Dla commit: 154528e508e5b6e9fb046bf603dd0959545b69c2 zmieniłem sposób modyfikowania aktywacji klasyfikatora w taki sposób jak opisane wyżej. Udało mi się w ten sposób poprawić skuteczność dla klasy "2", ale mam podejrzenia że tam jest coś nie tak. Bo normalnie powinno się odejmować ten kierunek odpowiedzialny za klasyfikację po kolorze, żeby był on mniej znaczący. W moim przypadku kiedy odejmowałem ten wektor to dostawałem gorsze wyniki i dopiero kiedy zacząłem dodawać to zacząłem dostawać lepsze (bez straty skuteczności dla innych klas). Trzeba by to lepiej przeanalizować, ale jest to też jakiś dowód, że poprzez modyfikację wartości aktywacji klasyfikatora da się poprawiać wyniki w konkretnych przypadkach.