if __name__ == '__main__':
    tot = int(input('total '))
    en = int(input('en '))
    lt = int(input('lt '))
    while True:
        coffs = input('Coffs/// en_n en_p lt_n lt_p > ')
        coffs = [float(x) for x in coffs.split(' ')]

        sc = [0, 0, 0, 0]
        sc[0] = coffs[0] * (tot - en)
        sc[1] = coffs[1] * en
        sc[2] = coffs[2] * (tot - lt)
        sc[3] = coffs[3] * lt
        print()
        for i in range(4):
            desc = 'en_n en_p lt_n lt_p'.split(' ')[i]
            print(f'{desc} {sc[i]} ({100*sc[i]/sum(sc):.2f}%)')
        print()
