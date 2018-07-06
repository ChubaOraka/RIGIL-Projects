package com.rigil.newtest;

import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.view.ViewTreeObserver;

import com.plattysoft.leonids.ParticleSystem;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
//        setSupportActionBar(toolbar);
//
//        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
//        fab.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
//                        .setAction("Action", null).show();
//            }
//        });
    }

//    @Override
//    public boolean onCreateOptionsMenu(Menu menu) {
//        // Inflate the menu; this adds items to the action bar if it is present.
//        getMenuInflater().inflate(R.menu.menu_main, menu);
//        return true;
//    }
//
//    @Override
//    public boolean onOptionsItemSelected(MenuItem item) {
//        // Handle action bar item clicks here. The action bar will
//        // automatically handle clicks on the Home/Up button, so long
//        // as you specify a parent activity in AndroidManifest.xml.
//        int id = item.getItemId();
//
//        //noinspection SimplifiableIfStatement
//        if (id == R.id.action_settings) {
//            return true;
//        }
//
//        return super.onOptionsItemSelected(item);
//    }

    @Override
    protected void onStart() {
        super.onStart();

//        ScrollView scroll = findViewById(R.id.scroll);
//        ParticleSystem ps = new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
//                .setSpeedRange(0.2f, 0.5f)
//                .oneShot(scroll, 500);
//        new ParticleSystem(this, 1000, R.drawable.ic_launcher_background, 100)
//                .setSpeedRange(0.2f, 0.5f)
//                .oneShot(scroll, 500);


//        new ParticleSystem(this, 80, R.drawable.star_white, 1000)
//                .setSpeedModuleAndAngleRange(0f, 0.3f, 180, 180)
//                .setRotationSpeed(144)
//                .setAcceleration(0.00005f, 90)
//                .emit(findViewById(R.id.helloTextView), 8);

        final View emiter_center = findViewById(R.id.emiter_center);
        final ViewTreeObserver viewTreeObserver = emiter_center.getViewTreeObserver();
        if (viewTreeObserver.isAlive()) {
            viewTreeObserver.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
                @Override
                public void onGlobalLayout() {
//                    viewTreeObserver.removeOnGlobalLayoutListener();
                    new ParticleSystem(MainActivity.this, 5, R.drawable.star_white, 3000)
                            .setSpeedModuleAndAngleRange(0f, 0.2f, 0, 180)
//                            .setSpeedByComponentsRange(-0.1f, 0.1f, -0.1f, 0.02f)
//                            .setInitialRotationRange(0, 180)
                            .setRotationSpeed(144)
                            .setAcceleration(0.000001f, 90)
//                            .setFadeOut(2000)
                            .emit(emiter_center, 2, 1500);
                }
            });
        }
//        new ParticleSystem(this, 80, R.drawable.star_white, 1000)
//                .setSpeedModuleAndAngleRange(0f, 0.3f, 180, 180)
//                .setRotationSpeed(144)
//                .setAcceleration(0.00005f, 90)
//                .emit(findViewById(R.id.emiter_top_right), 8);
//
//        new ParticleSystem(this, 80, R.drawable.star_white, 1000)
//                .setSpeedModuleAndAngleRange(0f, 0.3f, 0, 0)
//                .setRotationSpeed(144)
//                .setAcceleration(0.00005f, 90)
//                .emit(findViewById(R.id.emiter_top_left), 8);

    }
}
