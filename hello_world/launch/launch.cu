#include <iostream>
#include "../neo/neo.cuh"

using namespace std;

void print_array(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << "\n";
}

int main()
{
    int n = 10;
    int a = 2;

    int *x = new int[n];
    int *y = new int[n];
    int *z = new int[n];

    for (int i = 0; i < n; i++)
    {
        x[i] = i + 1;
        y[i] = i + 2;
    }
    print_array(x, n);
    print_array(y, n);

    affine::affine_transformer(a, x, y, z, n);

    for (int i = 0; i < n; i++)
        cout << z[i] << " ";
    cout << '\n';

    return 0;
}
