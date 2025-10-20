#include <iostream>
#include <vector>
#include <functional>
#include <execution>
#include <algorithm>
#include "timeit"
#include "random.h"

using namespace std;

vector<int> generateRandomSequence( int length )
{
    vector<int> v(length);
    std::uniform_int_distribution<int> distribution(-1e6, 1e6);
    for ( int i = 0; i < length; ++i )
    {
        v[i] = distribution(Random::engine());
    }
    return v;
}

template <typename Iterator>
using ValueType = typename std::iterator_traits<Iterator>::value_type;

template <typename Iterator, typename BinaryOperator>
ValueType<Iterator> MyParalelReduce( Iterator begin, Iterator end,
                     ValueType<Iterator> initialSum, BinaryOperator f, int threadsNumber )
{
    if ( threadsNumber == 1 )
        return reduce(execution::seq, begin, end, initialSum, f );

    vector< thread > threads;
    vector< ValueType<Iterator> > sums(threadsNumber);
    int chunkSize = distance(begin, end) / threadsNumber;
    if ( chunkSize < 1 )
        chunkSize = 1;

    int threadsCreated = 0;
    for ( int i = 0; i < threadsNumber; ++i )
    {
        auto chunkStart = next( begin, i * chunkSize );
        if ( chunkStart >= end )
            break;
        auto chunkEnd = next(chunkStart, chunkSize);
        if ( i == threadsNumber - 1 )
            chunkEnd = end;
        threads.emplace_back([ chunkStart, chunkEnd, &sums, i, f ](){
            sums[i] = reduce( execution::seq, next(chunkStart), chunkEnd, *chunkStart, f);
        });
        ++threadsCreated;
    }
    sums.resize(threadsCreated);
    for ( auto& t : threads ){
        t.join();
    }
    return reduce( execution::seq, sums.begin(), sums.end(), initialSum, f );
}

void test( const vector<int>& numbers )
{
    int repeats = 1000;
    if ( numbers.size() >= 1e6 )
        repeats = 100;
    
    cout << "Normal reduce" << endl;
    auto f = [](int a, int b)
        { return a + b; };
    cout << "no policy               ";
    timeit([&numbers, &f]()
        { return reduce(numbers.begin(), numbers.end(), 0, f); }, repeats);
    cout << "seq                     ";
    timeit([&numbers, &f]()
        { return reduce(execution::seq, numbers.begin(), numbers.end(), 0, f); }, repeats);
    cout << "parallel                ";
    timeit([&numbers, &f]()
        { return reduce(execution::par, numbers.begin(), numbers.end(), 0, f); }, repeats);
    cout << "unsequenced             ";
    timeit([&numbers, &f]()
        { return reduce(execution::unseq, numbers.begin(), numbers.end(), 0, f); }, repeats);
    cout << "parallel unsequenced    ";
    timeit([&numbers, &f]()
        { return reduce(execution::par_unseq, numbers.begin(), numbers.end(), 0, f); }, repeats);
    cout << "My parallel algorithm   " << endl;
    for ( int i = 2; i <= std::thread::hardware_concurrency() * 2; ++i )
    {
        cout << "   for " << i << " threads:   ";
        timeit([&numbers, &f, i]()
            { return MyParalelReduce(numbers.begin(), numbers.end(), 0, f, i); }, repeats);
    }

    cout << endl << "Reduce with heavy binary operation" << endl;
    auto heavyOp = [](int a, int b)
    {
        int temp = 0;
        for ( int i = 0; i < 5; ++i )
            temp += (a * i + b) % 1000000;
        return a + b + temp;
    };
    cout << "no policy               ";
    timeit([&numbers, &heavyOp]()
        { return reduce(numbers.begin(), numbers.end(), 0, heavyOp); }, repeats);
    cout << "seq                     ";
    timeit([&numbers, &heavyOp]()
        { return reduce(execution::seq, numbers.begin(), numbers.end(), 0, heavyOp); }, repeats);
    cout << "parallel                ";
    timeit([&numbers, &heavyOp]()
        { return reduce(execution::par, numbers.begin(), numbers.end(), 0, heavyOp); }, repeats);
    cout << "unsequenced             ";
    timeit([&numbers, &heavyOp]()
        { return reduce(execution::unseq, numbers.begin(), numbers.end(), 0, heavyOp); }, repeats);
    cout << "parallel unsequenced    ";
    timeit([&numbers, &heavyOp]()
        { return reduce(execution::par_unseq, numbers.begin(), numbers.end(), 0, heavyOp); }, repeats);
    cout << "My parallel algorithm   " << endl;
    for ( int i = 2; i <= std::thread::hardware_concurrency() * 2; ++i )
    {
        cout << "   for " << i << " threads:   ";
        timeit([&numbers, &heavyOp, i]()
            { return MyParalelReduce(numbers.begin(), numbers.end(), 0, heavyOp, i); }, repeats);
    }
    cout << endl;
}

int main()
{
    auto numbers = generateRandomSequence( 10000 );
    cout << "Test sequence with length 10000" << endl;
    test(numbers);
    numbers = generateRandomSequence( 1000000 );
    cout << "Test sequence with length 1000000" << endl;
    test(numbers);
    numbers = generateRandomSequence( 100000000 );
    cout << "Test sequence with length 100000000" << endl;
    test(numbers);
}
