import json

from rich import print
from rich.console import Console
from rich.padding import Padding
from rich.prompt import *
from sympy import Matrix, Rational, eye, symbols

console = Console()


def mensaje_padding(message):
    """
    Colorear mensajes
    """
    theme = "italic blue on white"
    length = len(message)
    mensaje = Padding(f"> {message}", (0, 15), style=theme, expand=True)
    return mensaje


def input_positive_integer(prompt):
    """
    Tomar valores del usuario
    """
    while True:
        # value = int(input(prompt))
        value = IntPrompt.ask(prompt)
        try:
            if value > 0:
                return value
            else:
                text = Text()
                text.append("Ingrese un numero positivo", style="red")
                console.print(text)

        except ValueError:
            text = Text()
            text.append("Porfavor ingrese un numero", style="red")
            console.print(text)


def input_matrix():
    rows = input_positive_integer("- numero de filas")
    cols = input_positive_integer("- numero de columnas")

    # matriz rellena con "_"
    a = symbols("_")
    matrix = [[a] * cols for _ in range(rows)]
    x = Matrix(matrix)
    console.clear()
    console.print(f"\nSu matriz ({rows}x{cols}):", style="white on blue")
    print_matrix(x)

    # matrix = []
    for i in range(rows):
        for j in range(cols):
            while True:
                val = input(f"Ingrese elemento en la pos ({i+1}, {j+1}): ")
                if val.lstrip("-").isdigit():
                    matrix[i][j] = Rational(int(val))
                    x = Matrix(matrix)
                    console.clear()
                    console.print(
                        f"\nSu matriz ({rows}x{cols}):", style="white on blue"
                    )
                    print_matrix(x, highlight_pos=(i, j))
                    break
                else:
                    console.print("Porfavor ingrese un numero.", style="red")

    console.clear()
    console.print(f"\nSu matriz ({rows}x{cols}):", style="white on blue")
    print_matrix(x)

    return x


def calculate_rank(x):
    return x.rank()


def print_matrix(x, affected_rows=None, is_augmented=False, highlight_pos=None):
    if isinstance(x, list):
        m, n = len(x), len(x[0])
        x = [[str(val) for val in row] for row in x]
    else:
        m, n = x.shape
        x = [[str(val) for val in row] for row in x.tolist()]

    # buscar el ancho de cada fila
    max_widths = [max(len(x[i][j]) for i in range(m)) for j in range(n)]

    for i in range(m):
        row = x[i]
        color = "red" if affected_rows and i in affected_rows else "white"
        row_str = []
        for j, val in enumerate(row):
            if is_augmented and j == m:
                row_str.append("|")
            if highlight_pos and highlight_pos == (i, j):
                color = "orange"
            row_str.append(f"[{color}]{val.rjust(max_widths[j])}[/{color}]")
        # console.print("[" + ", ".join(row_str) + "]")
        console.print("[" + ", ".join(row_str).replace("|,", "|") + "]")


# def escalonar_simple(x):
#     m, n = x.shape
#     # Convert the matrix to fractions
#     x = Matrix([[Rational(val) for val in row] for row in x.tolist()])
#
#     print("# Matriz original")
#     print_matrix(x)
#
#     i = 0
#     while i < min(m, n):
#         # Check for zero pivot element
#         if x[i, i] == 0:
#             # Find a row below with non-zero element in the same column
#             for j in range(i + 1, m):
#                 if x[j, i] != 0:
#                     # Swap rows
#                     x.row_swap(i, j)
#                     print(f"\n# Intercambie fila {i+1} con la fila {j+1}")
#                     print_matrix(x)
#
#             else:
#                 i += 1
#                 continue
#         for j in range(i + 1, m):
#             k = (-1) * x[j, i] / x[i, i]
#             if k != 0:
#                 x[j, :] = x[j, :] + k * x[i, :]
#                 print(f"\n# Sume {k} veces fila {i+1} a la fila {j+1}")
#                 print_matrix(x, affected_rows=[j])
#         i += 1
#
#     rango = calculate_rank(x)
#     console.print("\n# Matriz escalonada:", style="italic white on blue")
#     print_matrix(x)
#     print("\n# El rango de la matriz escalonada es:", rango)
#     return x


def gauss_jordan_inverse(a):
    """
    Buscamos la inversa de la matriz
    """
    m, n = a.shape
    x = a.copy()

    # Creamos la matriz aumentada
    x = x.row_join(eye(m))

    console.clear()
    print("\n# Matriz ampliada:")
    print_matrix(x, is_augmented=True)

    for i in range(min(m, n)):
        # Buscando ceros en los pivotes
        if x[i, i] == 0:
            # Intercambia con una fila sin pivote=0
            for j in range(i + 1, m):
                if x[j, i] != 0:
                    # intercambio
                    x.row_swap(i, j)
                    print(f"\n# Intercambie fila {i+1} con fila {j+1}")
                    print_matrix(x, affected_rows=[i], is_augmented=True)
                    break

        for j in range(i + 1, m):
            k = (-1) * x[j, i] / x[i, i]
            if k != 0:
                x[j, :] = x[j, :] + k * x[i, :]
                print(f"\n# f{j+1} = f{j+1} + ({k})f{i+1}")
                # print("\n# Sume {k} veces fila {i+1} a la fila {j+1}")
                print_matrix(x, affected_rows=[j], is_augmented=True)

    for i in range(m - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            k = (-1) * x[j, i] / x[i, i]
            if k != 0:
                x[j, :] = x[j, :] + k * x[i, :]
                # print(f"\n# Sume {k} veces fila {i+1}  a la fila {j+1}")
                print(f"\n# f{j+1} = f{j+1} + ({k})f{i+1}")
                print_matrix(x, affected_rows=[j], is_augmented=True)

    for i in range(m):
        # hacer el pivote = 1
        scalar = x[i, i]
        x[i, :] = x[i, :] / scalar
        # print(f"\n# Multiplique la fila {i+1} por {scalar**-1}")
        print(f"\n# f{i+1} = ({scalar**-1})f{i+1}")
        print_matrix(x, affected_rows=[i], is_augmented=True)

    return x[:, m:]


def row_echelon_form(a, solve_system=False):
    m, n = a.shape
    x = a.copy()

    print("\n# Matriz original")
    print_matrix(a, is_augmented=True)

    i = 0
    while i < min(m, n):
        # Buscando ceros en los pivotes
        if x[i, i] == 0:
            # intercambio de filas
            for j in range(i + 1, m):
                if x[j, i] != 0:
                    # intercambiamos
                    x.row_swap(i, j)
                    print(f"\n# Intercambie fila {i+1} con fila {j+1}")
                    print_matrix(x, affected_rows=[i])
                    break
            else:
                i += 1
                continue

        # Hacer el pivote 1
        pivot = x[i, i]
        if pivot != 1:  # saltar si ya es 1
            x[i, :] = x[i, :] / pivot
            print(f"\n# f{i+1} = ({pivot**-1})f{i+1}")
            # print(f"\n# Multiplique la fila {i+1} por {pivot**-1}")
            print_matrix(x, affected_rows=[i])

        for j in range(m):
            # saltar paso si hay 0
            if j != i and not all(x[j, k] == 0 for k in range(n)):
                k = (-1) * x[j, i] / x[i, i]
                if k != 0:
                    x[j, :] = x[j, :] + k * x[i, :]
                    print(f"\n# f{j+1} = f{j+1} + ({k})f{i+1}")
                    # print(f"\n# Sume {k} veces fila {i+1} a la fila {j+1}")
                    print_matrix(x, affected_rows=[j])
        i += 1

    rango = calculate_rank(x)
    console.print("\n# Matriz escalonada:", style="italic white on blue")
    print_matrix(x)
    print("\n# El rango de la matriz escalonada es:", rango)

    if solve_system:
        print("\n# Soluciones del sistema:")
        for i in range(min(m, n)):
            if i < m:
                console.print(f"- x_{i+1} = {x[i, n-1]}", style="bold")
            else:
                console.print(f"- x_{i+1} es una variable libre")


def input_augmented_matrix(matrix):
    rows, cols = matrix.shape
    console.clear()
    mensaje = mensaje_padding("Resolver sistema")
    console.print(mensaje)

    # ingresar valores
    a = symbols("_")
    augmented_matrix = [a] * rows
    matrix = matrix.row_join(Matrix(augmented_matrix))
    print_matrix(matrix, is_augmented=True)

    for i in range(rows):
        while True:
            val = input(f"Ingrese el valor para la fila {i+1} : ")
            if not val.lstrip("-").isdigit():
                symbol = symbols(val)
                augmented_matrix[i] = symbol
            else:  # val.lstrip("-").isdigit():
                augmented_matrix[i] = Rational(int(val))
            matrix = matrix[:, :-1].row_join(Matrix(augmented_matrix))
            console.clear()
            console.print(mensaje)
            print_matrix(matrix, is_augmented=True, affected_rows=[i])
            break

    return matrix


def print_matrix_and_inverse(x):
    # sacar inversa
    inverse = gauss_jordan_inverse(x)

    # pasar a lista
    x = [[str(val) for val in row] for row in x.tolist()]
    inverse = [[str(val) for val in row] for row in inverse.tolist()]

    # sacar el ancho
    max_widths_x = [max(len(x[i][j]) for i in range(len(x))) for j in range(len(x[0]))]
    max_widths_inverse = [
        max(len(inverse[i][j]) for i in range(len(inverse)))
        for j in range(len(inverse[0]))
    ]

    console.print("\n# Resultado final:", style="italic white on blue")
    for i in range(len(x)):
        row_x = [val.rjust(max_widths_x[j]) for j, val in enumerate(x[i])]
        row_inverse = [
            val.rjust(max_widths_inverse[j]) for j, val in enumerate(inverse[i])
        ]
        if i == 0:
            console.print(
                "[" + ", ".join(row_x) + "]^-1  [" + ", ".join(row_inverse) + "]"
            )
        elif i == len(x) // 2:
            console.print(
                "[" + ", ".join(row_x) + "]   = [" + ", ".join(row_inverse) + "]"
            )
        else:
            console.print(
                "[" + ", ".join(row_x) + "]     [" + ", ".join(row_inverse) + "]"
            )


def matrix_from_file(theme_2):
    console.clear()
    filename = "matrix.json"
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            matrices = data["matrices"]
    except FileNotFoundError:
        print("Archivo no encontrado", style="red")

    archivo = mensaje_padding(filename)
    print(archivo)

    print("Elija una matriz:")
    for i, x in enumerate(matrices):
        print(f"{i+1}. Matriz con dimension {len(x)}x{len(x[0])}")
    choices = [str(i + 1) for i in range(len(matrices))]
    matrix_choice = Prompt.ask("Elija una opcion", choices=choices)
    matrix_choice = int(matrix_choice.split(".")[0]) - 1
    x = Matrix(matrices[matrix_choice])

    console.clear()
    print("\n")
    console.print(f"Su matriz ({x.shape[0]}x{x.shape[1]}):", style=theme_2)
    print_matrix(x)

    return x


def main():
    console.clear()
    theme_2 = "white on blue"
    while True:
        console.clear()
        welcome = mensaje_padding("Matrices")
        print(welcome)
        print("1. Insertar matriz manualmente")
        print("2. Leer matrices desde archivo")
        print("3. Salir")
        choice = Prompt.ask("Elija una opcion", choices=["1", "2", "3"])

        if choice == "3":
            console.clear()
            break

        if choice == "1":
            console.clear()
            manual = mensaje_padding("Ingreso manual")
            console.print(manual)
            x = input_matrix()

        elif choice == "2":
            x = matrix_from_file(theme_2)

        else:
            print("Opcion invalida")
            continue

        while True:
            operaciones = mensaje_padding("Operaciones")
            print("\n", operaciones)
            print("1. Escalonar matriz")
            print("2. Buscar la inversa de la matriz")
            print("3. Resolver sistema")
            print("4. Volver al menu principal")

            operation_choice = Prompt.ask(
                "Elija una opcion", choices=["1", "2", "3", "4"]
            )

            if operation_choice == "4":  # Salir
                break

            elif operation_choice == "1":
                row_echelon_form(x)

            elif operation_choice == "2":
                if x.shape[0] != x.shape[1]:
                    console.clear()
                    console.print(
                        Text.assemble("\n# La matriz ", ("no", "red"), " es cuadrada!")
                    )
                else:
                    d = x.det()
                    if d == 0:
                        console.clear()
                        console.print(
                            Text.assemble(
                                f"\n# La determinante es {d}, ",
                                ("no", "red"),
                                " tiene inversa!",
                            )
                        )
                    else:
                        print_matrix_and_inverse(x)

            elif operation_choice == "3":
                x = input_augmented_matrix(x)
                row_echelon_form(x, solve_system=True)


if __name__ == "__main__":
    main()

