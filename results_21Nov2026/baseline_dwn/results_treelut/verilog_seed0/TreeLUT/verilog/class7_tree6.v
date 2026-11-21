module class7_tree6(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[406]) & (~i[158]) & (~i[385]) & i[355]) | (i[430] & i[453] & i[428] & (~i[355])) | ((~i[251]) & i[116] & i[79] & i[385] & i[355]) | ((~i[357]) & (~i[116]) & i[79] & i[385] & i[355]) | (i[265] & (~i[364]) & (~i[79]) & i[385] & i[355]) | (i[93] & i[467] & i[158] & (~i[385]) & i[355]) | ((~i[89]) & (~i[430]) & i[453] & i[428] & (~i[355])))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
