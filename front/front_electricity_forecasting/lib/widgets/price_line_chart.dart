import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class PriceLineChart extends StatelessWidget {
  final List<double> prices;

  const PriceLineChart({super.key, required this.prices});

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1.6,
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: LineChart(
          LineChartData(
            backgroundColor: Colors.white,

            // ✅ Només línies horitzontals
            gridData: FlGridData(
              show: true,
              drawHorizontalLine: true,
              drawVerticalLine: false,
              getDrawingHorizontalLine: (value) => const FlLine(
                color: Colors.black12,
                strokeWidth: 1,
              ),
            ),

            // ✅ Títols dels eixos
            titlesData: FlTitlesData(
              bottomTitles: AxisTitles(
                axisNameSize: 32,
                axisNameWidget: const Text(
                  'Hora del dia',
                  style: TextStyle(
                    color: Colors.black,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                sideTitles: SideTitles(
                  showTitles: true,
                  interval: 1,
                  reservedSize: 28,
                  getTitlesWidget: (value, _) {
                    final int hour = value.toInt();
                    if (hour < 0 || hour > 23) return const SizedBox();
                    return Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(
                        hour.toString(),
                        style: const TextStyle(fontSize: 12, color: Colors.black),
                      ),
                    );
                  },
                ),
              ),
              leftTitles: AxisTitles(
                axisNameSize: 32,
                axisNameWidget: const Padding(
                  padding: EdgeInsets.only(right: 8),
                  child: Text(
                    'Preu (€ / MWh)',
                    style: TextStyle(
                      color: Colors.black,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                sideTitles: SideTitles(
                  showTitles: true,
                  interval: 10,
                  reservedSize: 42,
                  getTitlesWidget: (value, _) => Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: Text(
                      '${value.toInt()}',
                      style: const TextStyle(fontSize: 12, color: Colors.black),
                    ),
                  ),
                ),
              ),
              rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
              topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
            ),

            // ✅ Tooltip personalitzat
            lineTouchData: LineTouchData(
              handleBuiltInTouches: true,
              touchTooltipData: LineTouchTooltipData(
                getTooltipColor: (touchedSpot) => Colors.black,
                // tooltipBgColor: Colors.black,
                // tooltipRoundedRadius: 8,
                fitInsideHorizontally: true,
                fitInsideVertically: true,
                getTooltipItems: (touchedSpots) {
                  return touchedSpots.map((spot) {
                    final hour = spot.x.toInt();
                    final price = spot.y.toStringAsFixed(2);
                    return LineTooltipItem(
                      'Hora $hour\n$price €/MWh',
                      const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    );
                  }).toList();
                },
              ),
            ),

            // ✅ Frontera
            borderData: FlBorderData(
              show: true,
              border: const Border(
                left: BorderSide(color: Colors.black),
                bottom: BorderSide(color: Colors.black),
              ),
            ),

            // ✅ Dades de la línia
            lineBarsData: [
              LineChartBarData(
                spots: List.generate(
                  prices.length,
                  (i) => FlSpot(i.toDouble(), prices[i]),
                ),
                isCurved: true,
                color: Colors.black,
                barWidth: 3,
                isStrokeCapRound: true,
                dotData: FlDotData(
                  show: true,
                  getDotPainter: (_, __, ___, ____) => FlDotCirclePainter(
                    radius: 3,
                    color: Colors.black,
                    strokeWidth: 0,
                  ),
                ),
              ),
            ],

            minX: 0,
            maxX: 23,
            minY: 0,
            maxY: (prices.reduce((a, b) => a > b ? a : b) + 10).ceilToDouble(),
          ),
        ),
      ),
    );
  }
}
