
import 'package:flutter/material.dart';
import 'package:front_electricity_forecasting/widgets/card_container.dart';

class DayFeaturesCard extends StatelessWidget {

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
    "is_mond",
    "is_tues",
    "is_wed",
    "is_thurs",
    "is_fri",
    "is_sat",
    "is_sun",
    "is_sunday_or_holiday",
    "hour_sin",
    "hour_cos",
    "type_day_workday",
    "type_day_sat",
    "type_day_sun",
    "type_day_holiday",
    "holiday_coef",
    "demand",
    "low_demand",
    "solar_share_demand",
    "wind_share_demand",
    "gas_generation_share",
    "gas_price",
    "residual_demand",
    "interchange_balance",
    "renewable_ratio",
    "high_renewable_ratio",
    "temp_dev",
    "price_es_24h",
    "renewables_to_gas",
    "demand_per_gas",
    "price_rolling_3h",
    "gas_price_lag1"
    ];

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: CardContainer(
        child: Column(
          children: [
            Text(
              'Hora $hour',
              style: TextStyle(
                fontSize: 24,
                color: Colors.black,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Divider(),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  "Preu Previst: ",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  "${preuPredit.toStringAsFixed(2)} €/MWh",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
      
            const Divider(),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  "MAE: ",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                  Text(
                  "${mae.toStringAsFixed(2)} €/MWh",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 20),
      
                Text(
                  "RMSE: ",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                  Text(
                  "${rmse.toStringAsFixed(2)} €/MWh",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 20),
      
                Text(
                  "SMAPE: ",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                  Text(
                  "${smape.toStringAsFixed(2)} %",
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 20),
              ],
            ),
      
            const Divider(),
            const SizedBox(height: 10),
      
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    for (int i = 0; i < featuresNames.length ~/ 2; i++)
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 2.0),
                        child: RichText(
                          text: TextSpan(
                            text: '${featuresNames[i]}: ',
                            style: TextStyle(
                              color: Colors.black,
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                            ),
                            children: [
                              TextSpan(
                                text: features[i].toString(),
                                style: TextStyle(
                                  color: Colors.black,
                                  fontWeight: FontWeight.normal,
                                  fontSize: 14,
                                ),
                              )
                            ]
                          ),
                        )
                      )
                  ],
                ),
                
                const SizedBox(width: 30),
      
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    for (int i = featuresNames.length ~/ 2 + 1; i < featuresNames.length; i++)
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 2.0),
                        child: RichText(
                          text: TextSpan(
                            text: '${featuresNames[i]}: ',
                            style: TextStyle(
                              color: Colors.black,
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                            ),
                            children: [
                              TextSpan(
                                text: features[i].toString(),
                                style: TextStyle(
                                  color: Colors.black,
                                  fontWeight: FontWeight.normal,
                                  fontSize: 14,
                                ),
                              )
                            ]
                          ),
                        )
                      )
                  ],
                )
              ],
            )
          ],
        ),
      ),
    );
  }

  
}