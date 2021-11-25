from typing import List

class Expression:
    def __init__(self, expression=None) -> None:
        self.expression : List[str] = []
        if isinstance(expression, list):
            self.expression = expression
        elif isinstance(expression, str):
            self.expression = expression.split('+')

    def __add__(self, rhs):
        return Expression(self.expression + rhs.expression)

    def __mul__(self, rhs):
        return Expression([term0 + term1 for term1 in rhs.expression for term0 in self.expression])
    
    def __repr__(self):
        return '+'.join(self.expression)


class Matrix:
    def __init__(self, rows) -> None:
        self.rows = [[Expression(x) if isinstance(x, str) else x for x in row] for row in rows]

    def __getitem__(self, key):
        return self.rows[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.rows[key[0]][key[1]] = value

    def __add__(self, rhs):
        return Matrix([[expression + rhs[i,j] for j, expression in enumerate(row)] for i, row in enumerate(self.rows)])

    def __mul__(self, rhs):
        if isinstance(rhs, Expression):
            return Matrix([x*rhs for x in row] for row in self.rows)
        else:
            new_rows = []
            columns = [[] for i, x in enumerate(rhs.rows[0])]
            for r, row in enumerate(rhs.rows):
                for c, x in enumerate(row):
                    columns[c].append(x)
            for row in self.rows:
                new_row = []
                for column in columns:
                    cell_value = Expression()
                    for i, elem in enumerate(column):
                        cell_value = cell_value + row[i]*column[i]
                    new_row.append(cell_value)
                new_rows.append(new_row)
            return Matrix(new_rows)
    
    def __repr__(self):
        return '\n'.join(['\t'.join([str(x) for x in row]) for row in self.rows])

    def convolve(self, other):
        new_rows = [[] for i in range(len(self.rows)+len(other.rows)-1)]
        return Matrix(new_rows)


x = Matrix([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']])
y = Matrix([['j', 'k', 'l'], ['m', 'n', 'o'], ['p', 'q', 'r']])
y = Matrix([['x'], ['y'], ['z']])
print(x)
print(y)
print(x*y)
x = Matrix([['a', 'b']])