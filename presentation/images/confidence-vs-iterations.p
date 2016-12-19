# set term x11 persist
set term png enhanced
set output "confidence-vs-iterations.png"

# set size ratio 1
set xlabel 'Iterations'
set ylabel 'GoogLeNet Confidence'
# set xrange [-6:6]
set yrange [-0.05:1.1]
# set xzeroaxis
# set yzeroaxis

plot 'confidence-vs-iterations.dat' using 1:2 lt rgb "#115EA6" linewidth 2 title 'Acorn', 'confidence-vs-iterations.dat' using 1:3 lt rgb "dark-green" linewidth 2 title 'Dresser'
