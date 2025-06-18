import 'package:flutter/material.dart';
import 'package:front_electricity_forecasting/providers/auth_provider.dart';
import 'package:front_electricity_forecasting/widgets/price_line_chart.dart';
import 'package:provider/provider.dart';

import 'package:front_electricity_forecasting/models/user.dart';
import 'package:front_electricity_forecasting/providers/predicts_provider.dart';
import 'package:front_electricity_forecasting/screens/login_screen.dart';
import 'package:front_electricity_forecasting/widgets/card_container.dart';
import 'package:front_electricity_forecasting/widgets/day_features_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late Future<bool> _fetchFuture;
  late User? user;


  @override
  void initState() {
    super.initState();
    user = Provider.of<AuthProvider>(context, listen: false).user;
    final predictsProvider = Provider.of<PredictsProvider>(context, listen: false);
    if (user == null) {
      // Si no hi ha user, redirigeix al login
      Future.microtask(() {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const LoginScreen()),
        );
      });
      return;
    }
    _fetchFuture = predictsProvider.fetch(user!.apiKey);
  }

  @override
  Widget build(BuildContext context) {
    final predictsProvider = Provider.of<PredictsProvider>(context);


    if (user == null) {
      // Si no hi ha user, redirigeix al login
      Future.microtask(() {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const LoginScreen()),
        );
      });
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 25),
          child: Text(
            'Prediccions del mercat elèctric',
            style: TextStyle(fontSize: 22, color: Colors.white),
          ),
        ),
        backgroundColor: Colors.black,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: () async {
              predictsProvider.isLoading = true;
              await predictsProvider.fetch(user!.apiKey);
              predictsProvider.isLoading = false;
            }
          ),
          const SizedBox(width: 30),
          IconButton(
            icon: const Icon(Icons.logout, color: Colors.white),
            onPressed: () {
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (_) => const LoginScreen()),
              );
            },
          ),
          const SizedBox(width: 30),
        ],
      ),
      body: FutureBuilder<bool>(
        future: _fetchFuture,
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return const Center(child: CircularProgressIndicator(color: Colors.black));
          }
          if (predictsProvider.isLoading) {
            return const Center(child: CircularProgressIndicator(color: Colors.black));
          }

          final maeDia = predictsProvider.prediction!.dailyMean.mae.toStringAsFixed(2);
          final rmseDia = predictsProvider.prediction!.dailyMean.rmse.toStringAsFixed(2);
          final smapeDia = predictsProvider.prediction!.dailyMean.smape.toStringAsFixed(2);

          final prices = predictsProvider.prediction!.hourPredictions.values.map((e) => e.predictedPrice).toList();

          return Padding(
            padding: const EdgeInsets.only(top: 30, left: 40, right: 40),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                //* Columna Esquerra
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 15),
                  // color: Colors.red,
                  width: MediaQuery.of(context).size.width * 0.4,
                  child: Column(
                    children: [

                      CardContainer(
                        width: double.infinity,
                        child: Column(
                          children: [
                            Text(
                              'Benvingut, ${user!.email}',
                              style: const TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.bold,
                                color: Colors.black,
                              ),
                            ),
                            const SizedBox(height: 20),
                            const Text(
                              'Pots utilitzar la següent API Key per accedir a les prediccions via API:',
                              style: TextStyle(fontSize: 18, color: Colors.black),
                            ),
                            const SizedBox(height: 10),
                            SelectableText(
                              user!.apiKey,
                              style: TextStyle(
                                fontSize: 18,
                                color: Colors.grey[700],
                                fontFamily: 'Courier',
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 40),

                      const Align(
                        alignment: Alignment.center,
                        child: Text(
                          "Features de la predicció",
                          style: TextStyle(fontSize: 26, color: Colors.black, fontWeight: FontWeight.bold),
                        ),
                      ),

                      const SizedBox(height: 20),

                      if (predictsProvider.features == null)
                        const Center(child: Text("No hi ha dades disponibles per a aquesta data", style: TextStyle(fontSize: 18, color: Colors.black)))
                      else
                        Expanded(
                          child: SingleChildScrollView(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: predictsProvider.features!.hours.entries.map((entry) {
                                final hour = entry.key;
                                final preuPredit = predictsProvider.prediction!.hourPredictions[hour]!.predictedPrice;
                                final mae = predictsProvider.prediction!.hourPredictions[hour]!.mae;
                                final rmse = predictsProvider.prediction!.hourPredictions[hour]!.rmse;
                                final smape = predictsProvider.prediction!.hourPredictions[hour]!.smape;
                                
                                return Padding(
                                  padding: const EdgeInsets.symmetric(vertical: 15),
                                  child: DayFeaturesCard(
                                    hour: hour,
                                    preuPredit: preuPredit,
                                    mae: mae,
                                    rmse: rmse,
                                    smape: smape,
                                    features: entry.value.features,
                                  ),
                                );
                              }).toList(),
                            ),
                          ),
                        )
                    ],
                  ),
                ),

                

                //* Columna Dreta
                Container(
                  padding: const EdgeInsets.only(bottom: 20, left: 30, right: 30),
                  // color: Colors.blue,
                  width: MediaQuery.of(context).size.width * 0.6 - 40 * 2,
                  child: Column(
                    children: [
                      CardContainer(
                        width: 660,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Container(
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: Colors.black,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Text(
                                predictsProvider.currentDate,
                                style: const TextStyle(
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                            const SizedBox(height: 20),
                            const Text(
                              'Mitjana de les mètriques de la predicció de cada hora:',
                              style: TextStyle(fontSize: 18, color: Colors.black),
                            ),
                            const SizedBox(height: 10),
                            
                            Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  "MAE Dia: ",
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                 Text(
                                  "$maeDia €/MWh",
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(width: 20),

                                Text(
                                  "RMSE Dia: ",
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                 Text(
                                  "$rmseDia €/MWh",
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(width: 20),

                                Text(
                                  "SMAPE Dia: ",
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                 Text(
                                  "$smapeDia %",
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(width: 20),
                              ],
                            )
                          ],
                        ),
                      ),

                      const SizedBox(height: 40),

                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10),
                        width: double.infinity,
                        height: 640,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: const [
                            BoxShadow(
                              color: Colors.black54,
                              blurRadius: 15,
                              offset: Offset(0, 4),
                            )
                          ],
                        ),
                        child: PriceLineChart(
                          prices: prices
                        ),
                      )
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}
