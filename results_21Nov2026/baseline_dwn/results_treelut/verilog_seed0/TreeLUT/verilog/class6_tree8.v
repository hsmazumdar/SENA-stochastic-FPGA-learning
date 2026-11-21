module class6_tree8(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[436])) | (i[245] & (~i[387]) & i[436]) | (i[31] & i[357] & i[338] & i[387] & i[436]) | (i[319] & i[292] & (~i[338]) & i[387] & i[436]) | ((~i[117]) & (~i[236]) & (~i[245]) & (~i[387]) & i[436]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
