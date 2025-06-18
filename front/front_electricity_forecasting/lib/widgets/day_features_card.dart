import 'package:flutter/material.dart';
import 'package:front_electricity_forecasting/widgets/card_container.dart';

class DayFeaturesCard extends StatefulWidget {
  final String hour;
  final double preuPredit;
  final double mae;
  final double rmse;
  final double smape;
  final List<double> features;

  const DayFeaturesCard({
    super.key,
    required this.hour,
    required this.features,
    required this.preuPredit,
    required this.mae,
    required this.rmse,
    required this.smape,
  });

  static const List<String> featuresNames = [
    "is_mond", "is_tues", "is_wed", "is_thurs", "is_fri", "is_sat", "is_sun", "is_sunday_or_holiday",
    "hour_sin", "hour_cos", "type_day_workday", "type_day_sat", "type_day_sun", "type_day_holiday",
    "holiday_coef", "demand", "low_demand", "solar_share_demand", "wind_share_demand",
    "gas_generation_share", "gas_price", "residual_demand", "interchange_balance",
    "renewable_ratio", "high_renewable_ratio", "temp_dev", "price_es_24h", "renewables_to_gas",
    "demand_per_gas", "price_rolling_3h", "gas_price_lag1"
  ];

  @override
  State<DayFeaturesCard> createState() => _DayFeaturesCardState();
}

class _DayFeaturesCardState extends State<DayFeaturesCard> with SingleTickerProviderStateMixin {
  bool _expanded = false;

  void _toggleExpanded() {
    setState(() {
      _expanded = !_expanded;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: _toggleExpanded,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          child: CardContainer(
            child: AnimatedSize(
              duration: const Duration(milliseconds: 300),
              curve: Curves.fastOutSlowIn,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Títol hora
                  Text(
                    'Hora ${widget.hour}',
                    style: const TextStyle(fontSize: 24, color: Colors.black, fontWeight: FontWeight.bold),
                  ),
                  const Divider(),
      
                  // Preu previst
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text("Preu Previst: ", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                      Text(
                        "${widget.preuPredit.toStringAsFixed(2)} €/MWh",
                        style: const TextStyle(fontSize: 16),
                      ),
                    ],
                  ),
      
                  const Divider(),
      
                  // Mètriques d'error
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      _metric("MAE", widget.mae, "€/MWh"),
                      const SizedBox(width: 20),
                      _metric("RMSE", widget.rmse, "€/MWh"),
                      const SizedBox(width: 20),
                      _metric("SMAPE", widget.smape, "%"),
                    ],
                  ),
      
                  const SizedBox(height: 10),
      
                  // Features expandibles
                  if (_expanded) ...[
                    const Divider(),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            for (int i = 0; i < widget.features.length ~/ 2; i++)
                              _featureText(i, widget.features[i])
                          ],
                        ),
                        const SizedBox(width: 40),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            for (int i = widget.features.length ~/ 2; i < widget.features.length; i++)
                              _featureText(i, widget.features[i])
                          ],
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _metric(String label, double value, String unit) {
    return Row(
      children: [
        Text("$label: ", style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        Text("${value.toStringAsFixed(2)} $unit", style: const TextStyle(fontSize: 16)),
      ],
    );
  }

  Widget _featureText(int index, double value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: RichText(
        text: TextSpan(
          text: '${DayFeaturesCard.featuresNames[index]}: ',
          style: const TextStyle(color: Colors.black, fontWeight: FontWeight.bold, fontSize: 14),
          children: [
            TextSpan(
              text: value.toString(),
              style: const TextStyle(color: Colors.black, fontWeight: FontWeight.normal),
            )
          ],
        ),
      ),
    );
  }
}
